
from __future__ import division
import math
import imutils
import scipy.misc
import numpy as np
import copy
import matplotlib.pyplot as plt
from random import *
import random
import cv2

def preprocess_image(dataA_list, patch, patch_size):
	imgA 	= [get_image(image_path=img_path, patch=patch, patch_size=patch_size) for img_path in dataA_list]
	imgA 	= np.array(imgA)
	return imgA

def get_image(image_path, patch=True, patch_size=128):
	img_A     	= cv2.imread(image_path)
	img_A 		= img_A / 255.0
	if patch == True:
		img_A 		= get_patch(img_A, patch_size)
	else:
		img_A 		= img_A
	img_A 		= data_aug(img_A)
	img_A   	= np.expand_dims(img_A, axis=2)
	return img_A

def load_test_data(image_path):
	img     		= cv2.imread(image_path)
	img 		= img / 255.0
	img 		= np.expand_dims(np.expand_dims(img, axis=2), axis=0)
	return img


def get_patch(img, patch_size):
	height      	= np.shape(img)[0]
	width       	= np.shape(img)[1]
	
	edge_height 	= height - patch_size
	if edge_height == 0 :
		random1     = 0
	else:
		random1     = random.randrange(0,edge_height)
	edge_width  	= width - patch_size
	if edge_width == 0 :
		random2     = 0
	else:
		random2     = random.randrange(0,edge_width)   
	
	output = img[random1 : random1 + patch_size, random2 : random2 + patch_size]

	return output    

def data_aug(img):
	rand_num    	= random.random()
	if rand_num > 0.7:
		img 		= np.flip(img, axis=0)
	rand_num    		= random.random()
	if rand_num > 0.7:
		img 		= np.flip(img, axis=1)
	rand_num    		= random.random()
	if rand_num > 0.7:
		rot_angle 	= random.randrange(1,179)
		img 		= imutils.rotate(img, angle=rot_angle)
	rand_num    	= random.random()
	if rand_num > 0.7:
		scale 		= random.uniform(0.9,1.1)
		img 		= scale * img
	return img

# -----------------------------
def save_images(real, fake, recon, image_size, file_name, num=4):
	img 	= np.concatenate((real, fake, recon), axis=0)
	img 	= make3d(img, image_size, row=num, col=3)
	cv2.imwrite(file_name, img)

def save_test_images(images, image_path):
	return cv2.imwrite(image_path, images)

def make3d(img, image_size, row, col):
	img 	= np.squeeze(img)
	img 	= np.reshape(img, [col, row, image_size, image_size, 1])  
	img 	= unstack(img, axis=0)  
	img 	= np.concatenate(img, axis=2)  
	img 	= unstack(img, axis=0)  
	img 	= np.concatenate(img, axis=0) 
	return img

def unstack(img, axis):
	d 	= img.shape[axis]
	arr 	= [np.squeeze(a, axis=axis) for a in np.split(img, d, axis=axis)]
	return arr
