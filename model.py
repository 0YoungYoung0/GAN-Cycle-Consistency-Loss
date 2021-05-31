from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *

class cyclegan(object):
	def __init__(self, sess, args):
		self.sess              	= sess
		self.batch_size 		= args.batch_size

		self.load_size 		= args.load_size
		self.patch_size 		= args.patch_size

		self.input_c_dim 	= args.input_nc
		self.output_c_dim 	= args.output_nc
		self.L1_lambda 		= args.L1_lambda
		self.iden_lambda 	= args.iden_lambda

		self.test_dir      	= args.dataset_dir + args.dataset_type + '/test_plane/'
		self.valid_dir 		= args.dataset_dir + args.dataset_type + '/valid_plane/'
		self.input_dir     	= args.dataset_dir + args.dataset_type + '/input_plane/'
		self.target_dir     	= args.dataset_dir + args.dataset_type + '/target_plane/'
		self.log_dir 		= args.log_dir
		self.checkpoint_dir 	= args.checkpoint_dir
		self.sample_dir 		= args.sample_dir
		self.save_dir 		= args.save_dir

		self.discriminator 		= discriminator
		if args.generator_type == 'generator_unet':
			self.generator 		= generator_unet
		elif args.generator_type == 'generator_resnet':
			self.generator 		= generator_resnet

		if args.use_lsgan:
			self.criterionGAN 	= mae_criterion
		else:
			self.criterionGAN 	= sce_criterion

		OPTIONS 			= namedtuple('OPTIONS', 'batch_size gf_dim df_dim output_c_dim is_training')
		self.options 			= OPTIONS._make((args.batch_size,  args.ngf, args.ndf, args.output_nc,  args.phase))

		self._build_model()
		self.saver 			= tf.train.Saver(max_to_keep=None)

	def _build_model(self):
		self.real_A		= tf.placeholder(tf.float32,[None, self.patch_size, self.patch_size, self.input_c_dim],name='real_A')
		self.real_B		= tf.placeholder(tf.float32,[None, self.patch_size, self.patch_size, self.input_c_dim],name='real_B')

		self.fake_B 		= self.generator(self.real_A, self.options, False, name="generatorA2B")
		self.fake_A_ 		= self.generator(self.fake_B, self.options, False, name="generatorB2A")
		self.fake_A 		= self.generator(self.real_B, self.options, True, name="generatorB2A")
		self.fake_B_ 		= self.generator(self.fake_A, self.options, True, name="generatorA2B")

		self.I_fake_B 		= self.generator(self.real_B, self.options, True, name="generatorA2B")
		self.I_fake_A 		= self.generator(self.real_A, self.options, True, name="generatorB2A")

		self.DA_real 		= self.discriminator(self.real_A, self.options, reuse=False, name="discriminatorA")
		self.DB_real 		= self.discriminator(self.real_B, self.options, reuse=False, name="discriminatorB")
		self.DA_fake 		= self.discriminator(self.fake_A, self.options, reuse=True, name="discriminatorA")
		self.DB_fake 		= self.discriminator(self.fake_B, self.options, reuse=True, name="discriminatorB")
		
		self.g_adv_loss 		= (self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake))) / 2.0
		self.g_recon_loss 	= self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) 
		self.g_iden_loss 	= self.iden_lambda * (abs_criterion(self.real_B, self.I_fake_B) + abs_criterion(self.real_A, self.I_fake_A))
		self.g_loss 		= self.g_adv_loss + self.g_recon_loss + self.g_iden_loss

		self.da_adv_loss 	= (self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real)) + self.criterionGAN(self.DA_fake, tf.zeros_like(self.DA_fake))) / 2.0
		self.db_adv_loss 	= (self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real)) + self.criterionGAN(self.DB_fake, tf.zeros_like(self.DB_fake))) / 2.0
		self.d_loss 		= self.da_adv_loss + self.db_adv_loss

		self.test_A         	= tf.placeholder(tf.float32,  [None, None, None,self.input_c_dim], name='test_A')
		self.test_fake_B       	= self.generator(self.test_A, self.options, True, name="generatorA2B")
		self.test_recon_A 	= self.generator(self.test_fake_B, self.options, True, name="generatorB2A")

		## Tensorboard 
		self.g_adv_sum 	= tf.summary.scalar("g_adv_loss", self.g_adv_loss)
		self.g_recon_sum 	= tf.summary.scalar("g_recon_loss", self.g_recon_loss)
		self.g_iden_sum 	= tf.summary.scalar("g_iden_loss", self.g_iden_loss)
		self.g_loss_sum 	= tf.summary.scalar("g_loss", self.g_loss)
		self.g_sum 		= tf.summary.merge([self.g_adv_sum, self.g_recon_sum, self.g_iden_sum, self.g_loss_sum])

		self.da_adv_sum 	= tf.summary.scalar("da_adv_loss", self.da_adv_loss)
		self.db_adv_sum 	= tf.summary.scalar("db_adv_loss", self.db_adv_loss)
		self.d_loss 		= tf.summary.scalar("d_loss", self.d_loss)
		self.d_sum 		= tf.summary.merge([self.da_adv_sum, self.db_adv_sum, self.d_loss])

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]

	def train(self, args):
		self.lr 		= tf.placeholder(tf.float32, None, name='learning_rate')
		self.d_optim 	= tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
		self.g_optim 	= tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

		init_op 		= tf.global_variables_initializer()
		self.sess.run(init_op)
		self.writer 	= tf.summary.FileWriter(self.log_dir, self.sess.graph)

		counter 	= 1
		start_time 	= time.time()

		if args.continue_train:
			if self.load(args.checkpoint_dir):
				print(" [*] Load SUCCESS")
			else:
				print(" [!] Load failed...")

		dataA 		= glob(os.path.join(self.input_dir + '*'))
		dataB 		= glob(os.path.join(self.target_dir + '*'))
		batch_idxs 	= min(len(dataA), len(dataB)) // self.batch_size
		for epoch in range(args.epoch):
			np.random.shuffle(dataA)
			np.random.shuffle(dataB)

			lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
			for idx in range(0, batch_idxs):
				data_list_input         		= data_files_input[idx * self.batch_size : (idx+1) * self.batch_size] 
				data_list_target        		= data_files_target[idx * self.batch_size : (idx+1) * self.batch_size]

				real_images            		= preprocess_image(data_list_input, patch=self.patch_train, patch_size=self.patch_size) 
				target_images  		= preprocess_image(data_list_target, patch=self.patch_train, patch_size=self.patch_size)

				feed 				= {self.real_A: real_images, self.real_B: target_images, self.lr: lr}
				_, _, g_summary, d_summary 	= self.sess.run([self.g_optim, self.d_optim, self.g_sum, self.d_sum],feed_dict=feed)
				self.writer.add_summary(g_summary, counter)
				self.writer.add_summary(d_summary, counter)

				counter += 1
				print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))

			if np.mod(epoch, args.print_freq) == 0:
				self.sample_model(epoch)
				self.checkpoint_save(epoch)

	def checkpoint_save(self, step):
		model_name = "model"
		self.saver.save(self.sess,os.path.join(self.checkpoint_dir, model_name),global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print ('found!',ckpt_name)
			self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
			return True
		else: 
			return False    

	def sample_model(self, epoch):
		test_list 			= glob(os.path.join(self.valid_dir + '*'))
		np.random.shuffle(test_list)

		test_list_input         		= test_list[idx * self.batch_size : (idx+1) * self.batch_size] 
		test_images            		= preprocess_image(test_list_input, patch=self.patch_train, patch_size=self.patch_size) 

		test_fake, test_recon		= self.sess.run([self.test_fake_B,self.test_recon_A],feed_dict={self.test_A: dataA1})

		test_images 			= test_images * 255.0
		test_fake_images 		= test_fake * 255.0
		test_recon_images 		= test_recon * 255.0
		
		save_file_name 		=  '{}/test_{:02d}.jpg'.format(self.sample_dir, epoch)
		save_images(test_images, test_fake_images, test_recon_images, self.load_size, save_file_name, num=self.batch_size)


	def test(self, args):
		self.load(self.checkpoint_dir)
		print(" [*] before training, Load SUCCESS ")            
		
		test_dir          		= glob(os.path.join(self.test_dir, '*'))

		data_list 		= os.listdir(self.test_dir)
		data_num 		= np.shape(data_list)[0]
		for da in range(data_num):
			print (da)
			data_name 	= data_list[da]
			test_full_path 	= os.path.join(self.test_dir, data_name)

			real_img	= load_test_data(test_full_path)

			fake_B 		= self.sess.run(self.test_fake_B,feed_dict={self.test_data: real_img})

			fake_B 		= np.squeeze(fake_B) * 255.0 

			save_full_path	= self.save_dir + '/'+ data_name
			save_test_images(fake_B, save_full_path)

		