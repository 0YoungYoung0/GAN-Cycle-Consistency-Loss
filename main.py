"""
Basic codes from "https://github.com/xhujoy/CycleGAN-tensorflow"
"""
import argparse
import os
import tensorflow as tf
tf.set_random_seed(20)
from model import cyclegan
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', 		dest='phase', 	default='train')
parser.add_argument('--dataset_dir',   		type=str, 	default='D:/DATASET') 
parser.add_argument('--generator_type', 	type=str, 	default='generator_unet',    help='generator_type') 
parser.add_argument('--dataset_type', 		type=str, 	default='dataset1',    help='dataset type')

parser.add_argument('--epoch', 		dest='epoch', type=int, default=100)
parser.add_argument('--epoch_step', 		dest='epoch_step', type=int, default=25)
parser.add_argument('--batch_size', 		dest='batch_size', type=int, default=1)
parser.add_argument('--load_size', 		dest='load_size', type=int, default=968)
parser.add_argument('--patch_size', 		dest='patch_size', type=int, default=256)

parser.add_argument('--ngf',			dest='ngf', type=int, default=32)
parser.add_argument('--ndf', 			dest='ndf', type=int, default=64)
parser.add_argument('--input_nc', 		dest='input_nc', type=int, default=1)
parser.add_argument('--output_nc', 		dest='output_nc', type=int, default=1)

parser.add_argument('--lr', 			dest='lr', type=float, default=1e-4)
parser.add_argument('--beta1', 		dest='beta1', type=float, default=0.5)
parser.add_argument('--print_freq', 		dest='print_freq', type=int, default=1)
parser.add_argument('--continue_train', 	dest='continue_train', type=bool, default=True)

parser.add_argument('--checkpoint_dir', 	dest='checkpoint_dir', default='checkpoint')
parser.add_argument('--sample_dir', 		dest='sample_dir', default='sample')
parser.add_argument('--save_dir', 		dest='save_dir', default='test')
parser.add_argument('--log_dir', 		dest='log_dir', default='log')

parser.add_argument('--L1_lambda', 		dest='L1_lambda', type=float, default=10.0)
parser.add_argument('--iden_lambda', 	dest='iden_lambda', type=float, default=5.0)
parser.add_argument('--use_lsgan', 		dest='use_lsgan', type=bool, default=True)

args = parser.parse_args()


def main(_):
	assets_dir              		= os.path.join('.','assets','load{}_patch{}_gen{}_ls{}_iden{}_lr{}_ep{}'.format(args.load_size, args.patch_size, args.generator_type, args.L1_lambda, args.iden_lambda, args.lr, args.epoch))

	args.checkpoint_dir         	= os.path.join(assets_dir, args.checkpoint_dir)
	args.sample_dir          		= os.path.join(assets_dir, args.sample_dir)
	args.save_dir       		= os.path.join(assets_dir, args.save_dir)
	args.log_dir          		= os.path.join(assets_dir, args.log_dir)
	
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)
	if not os.path.exists(args.sample_dir):
		os.makedirs(args.sample_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	tfconfig 				= tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth 	= True
	with tf.Session(config=tfconfig) as sess:
		model 				= cyclegan(sess, args)
		if args.phase == 'train':
			model.train(args) 
		elif args.phase == 'test':
			model.test(args)

if __name__ == '__main__':
	tf.app.run()
