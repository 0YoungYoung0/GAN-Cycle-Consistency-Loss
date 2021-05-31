from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def discriminator(image, options, reuse=False, name="discriminator"):

	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
		h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
		h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
		h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
		h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
		return h4

def generator_resnet(image, options, reuse=False, name="generator"):

	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		def residule_block(x, dim, ks=3, s=1, name='res'):
			p = int((ks - 1) / 2)
			y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
			y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
			y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
			y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
			return y + x
		c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
		c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
		c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

		r1      = residule_block(c3, options.gf_dim*4, name='g_r1')
		r2      = residule_block(r1, options.gf_dim*4, name='g_r2')
		r3      = residule_block(r2, options.gf_dim*4, name='g_r3')
		r4      = residule_block(r3, options.gf_dim*4, name='g_r4')
		r5      = residule_block(r4, options.gf_dim*4, name='g_r5')
		r6      = residule_block(r5, options.gf_dim*4, name='g_r6')
		r7      = residule_block(r6, options.gf_dim*4, name='g_r7')
		r8      = residule_block(r7, options.gf_dim*4, name='g_r8')
		r9      = residule_block(r8, options.gf_dim*4, name='g_r9')

		d1      = bilinear_deconv(r9, options.gf_dim*2, 3, 1, name='g_d1_dc')
		d1      = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
		d2      = bilinear_deconv(d1, options.gf_dim, 3, 1, name='g_d2_dc')
		d2      = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
		d2      = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		pred    = conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c')
		return pred

def generator_unet(image, options, reuse=False, name="generator"):

	if options.is_training == 'train':
		dropout_rate = 0.5 
	else:
		dropout_rate = 1.0
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		e11    	= instance_norm(conv2d(image, options.gf_dim, ks=3, s=1, name='g_e11_conv'), 'g_bn_e11')
		e12    	= instance_norm(conv2d(e11, options.gf_dim, ks=3, s=1, name='g_e12_conv'), 'g_bn_e12')
		e13    	= conv2d(e12, options.gf_dim, ks=3, s=2, name='g_e13_conv')
		
		e21    	= instance_norm(conv2d(lrelu(e13), options.gf_dim*2, ks=3, s=1, name='g_e21_conv'), 'g_bn_e21')
		e22    	= instance_norm(conv2d(lrelu(e21), options.gf_dim*2, ks=3, s=1, name='g_e22_conv'), 'g_bn_e22')
		e23    	= conv2d(lrelu(e22), options.gf_dim*2, ks=3, s=2, name='g_e23_conv')
		
		e31    	= instance_norm(conv2d(lrelu(e23), options.gf_dim*4, ks=3, s=1, name='g_e31_conv'), 'g_bn_e31')
		e32    	= instance_norm(conv2d(lrelu(e31), options.gf_dim*4, ks=3, s=1, name='g_e32_conv'), 'g_bn_e32')
		e33    	= conv2d(lrelu(e32), options.gf_dim*4, ks=3, s=2, name='g_e33_conv')
		
		e41    	= instance_norm(conv2d(lrelu(e33), options.gf_dim*8, ks=3, s=1, name='g_e41_conv'), 'g_bn_e41')
		e42    	= instance_norm(conv2d(lrelu(e41), options.gf_dim*8, ks=3, s=1, name='g_e42_conv'), 'g_bn_e42')   

		d1    	= bilinear_deconv(tf.nn.relu(e42), options.gf_dim*8, name='g_d1')
		d1     	= tf.concat([instance_norm(d1, 'g_bn_d1'), e32], 3)
		d1     	= tf.nn.dropout(d1, dropout_rate)
		d11    	= instance_norm(conv2d(lrelu(d1), options.gf_dim*8, ks=3, s=1, name='g_d11_conv'), 'g_bn_d11')
		d12    	= instance_norm(conv2d(lrelu(d11), options.gf_dim*4, ks=3, s=1, name='g_d12_conv'), 'g_bn_d12')

		d2     	= bilinear_deconv(tf.nn.relu(d12), options.gf_dim*4, name='g_d2')
		d2     	= tf.concat([instance_norm(d2, 'g_bn_d2'), e22], 3)
		d2     	= tf.nn.dropout(d2, dropout_rate)
		d21    	= instance_norm(conv2d(lrelu(d2), options.gf_dim*4, ks=3, s=1, name='g_d21_conv'), 'g_bn_d21')
		d22    	= instance_norm(conv2d(lrelu(d21), options.gf_dim*2, ks=3, s=1, name='g_d22_conv'), 'g_bn_d22')

		d3 	= bilinear_deconv(tf.nn.relu(d22), options.gf_dim*2, name='g_d3')
		d3 	= tf.concat([instance_norm(d3, 'g_bn_d3'), e12], 3)
		d3 	= tf.nn.dropout(d3, dropout_rate)
		d31 	= instance_norm(conv2d(lrelu(d3), options.gf_dim*2, ks=3, s=1, name='g_d31_conv'), 'g_bn_d31')
		d32 	= instance_norm(conv2d(lrelu(d31), options.gf_dim, ks=3, s=1, name='g_d32_conv'), 'g_bn_d32')
		
		pred 	= conv2d(lrelu(d32), options.output_c_dim, ks=1, s=1, name='pred_conv')

		return pred

def abs_criterion(in_, target):
	return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
	return tf.reduce_mean((in_-target)**2)
