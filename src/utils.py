"""
This file contains important utility functions used during training, validation and testing.

@author: Aditya Vora

"""

import glob
import os
import random
import numpy as np
import tensorflow as tf
import sys
import cv2



def get_density_map_gaussian(points, d_map_h, d_map_w):		#后两个参数预估需要和最后的前馈输出结果做比较	，从算法来看，应该是不用的
	"""
	Creates density maps from ground truth point locations
	:param points: [x,y] x: along width, y: along height
	:param d_map_h: height of the density map
	:param d_map_w: width of the density map
	:return: density map
	"""
	'''
	if d_map_h%4 == 0:
		d_map_h = int(0.25*d_map_h)
	else: d_map_h = int(0.25*d_map_h)+1	#根据same pooling进行长宽计算。
	if d_map_w%4 == 0:
		d_map_w = int(0.25*d_map_w)
	else: d_map_w = int(0.25*d_map_w)+1#用于将其尺寸在进行获取图像前就先进行缩放。
	'''
	d_map_h = 0.5*d_map_h
	if d_map_h!=int(d_map_h):
		d_map_h = int(d_map_h)+1
	d_map_h = 0.5*d_map_h
	if d_map_h!=int(d_map_h):
		d_map_h = int(d_map_h)+1#完成缩小到1/4
	d_map_w = 0.5*d_map_w
	if d_map_w!=int(d_map_w):
		d_map_w = int(d_map_w)+1
	d_map_w = 0.5*d_map_w
	if d_map_w!=int(d_map_w):
		d_map_w = int(d_map_w)+1#完成缩小到1/4
	d_map_h = int(d_map_h)
	d_map_w = int(d_map_w)

	im_density = np.zeros(shape=(d_map_h,d_map_w), dtype=np.float32)

	if np.shape(points)[0] == 0:  #如果输入的数据为空
		sys.exit()

	for i in range(np.shape(points)[0]):  #遍历每一条记录

		f_sz = 3.5#原来是15，既然要缩小整体图像大小，这里大概也缩一下好了。
		sigma = 4

		gaussian_kernel = get_gaussian_kernel(f_sz, f_sz, sigma)		 #得到fsize，长宽为15的高斯核
					
		x = min(d_map_w, max(1, np.abs(np.int32(np.floor(points[i, 0]/4)))))
		y = min(d_map_h, max(1, np.abs(np.int32(np.floor(points[i, 1]/4)))))

		if(int(x/4) > d_map_w or int(y/4) > d_map_h):
			continue

		x1 = x - np.int32(np.floor(f_sz / 2))
		y1 = y - np.int32(np.floor(f_sz / 2))
		x2 = x + np.int32(np.floor(f_sz / 2))
		y2 = y + np.int32(np.floor(f_sz / 2))

		dfx1 = 0
		dfy1 = 0
		dfx2 = 0
		dfy2 = 0

		change_H = False

		if(x1 < 1):
			dfx1 = np.abs(x1)+1
			x1 = 1
			change_H = True

		if(y1 < 1):
			dfy1 = np.abs(y1)+1
			y1 = 1
			change_H = True

		if(x2 > d_map_w):
			dfx2 = x2 - d_map_w
			x2 = d_map_w
			change_H = True

		if(y2 > d_map_h):
			dfy2 = y2 - d_map_h
			y2 = d_map_h
			change_H = True

		x1h = 1+dfx1
		y1h = 1+dfy1
		x2h = f_sz - dfx2
		y2h = f_sz - dfy2

		if (change_H == True):
			f_sz_y = np.double(y2h - y1h + 1)
			f_sz_x = np.double(x2h - x1h + 1)	#计算得到需要得到的卷积后的区域大小（用于卷积的区域大小）

			gaussian_kernel = get_gaussian_kernel(f_sz_x, f_sz_y, sigma)

		im_density[y1-1:y2,x1-1:x2] = im_density[y1-1:y2,x1-1:x2] +  gaussian_kernel
	return im_density

def get_gaussian_kernel(fs_x, fs_y, sigma):
	"""
	Create a 2D gaussian kernel
	:param fs_x: filter width along x axis
	:param fs_y: filter width along y axis
	:param sigma: gaussian width
	:return: 2D Gaussian filter of [fs_y x fs_x] dimension
	"""
	gaussian_kernel_x = cv2.getGaussianKernel(ksize=np.int(fs_x), sigma=sigma)
	gaussian_kernel_y = cv2.getGaussianKernel(ksize=np.int(fs_y), sigma=sigma)
	gaussian_kernel = gaussian_kernel_y * gaussian_kernel_x.T
	return gaussian_kernel

def compute_abs_err(pred, gt):
	"""
	Computes mean absolute error between the predicted density map and ground truth
	:param pred: predicted density map
	:param gt: ground truth density map
	:return: abs |pred - gt|
	"""
	return np.abs(np.sum(pred[:]) - np.sum(gt[:]))

def create_session(log_dir, session_id):
	"""
	Module to create a session folder. It will create a folder with a proper session
	id and return the session path.
	:param log_dir: root log directory
	:param session_id: ID of the session
	:return: path of the session id folder
	"""
	folder_path = os.path.join(log_dir, 'session-'+str(session_id))
	if os.path.exists(folder_path):
		print ('Session already taken. It will create a different session id.')#所以最好每次删了logs文件夹
		
		#sys.exit()
	else:
		os.makedirs(folder_path)
	return folder_path

def get_file_id(filepath):
	return os.path.splitext(os.path.basename(filepath))[0]

def get_data_list(data_root, mode='train'):

	"""
	Returns a list of images that are to be used during training, validation and testing.
	It looks into various folders depending on the mode and prepares the list.
	:param mode: selection of appropriate mode from train, validation and test.
	:return: a list of filenames of images and corresponding ground truths after random shuffling.
	"""

	if mode == 'train':
		imagepath = os.path.join(data_root, 'train_data', 'images')
		gtpath = os.path.join(data_root, 'train_data', 'ground_truth')

	elif mode == 'valid':
		imagepath = os.path.join(data_root, 'valid_data', 'images')
		gtpath = os.path.join(data_root, 'valid_data', 'ground_truth')

	else:
		imagepath = os.path.join(data_root, 'test_data', 'images')
		gtpath = os.path.join(data_root, 'test_data', 'ground_truth')

	image_list = [file for file in glob.glob(os.path.join(imagepath,'*.jpg'))]
	gt_list = []

	for filepath in image_list:
		file_id = get_file_id(filepath)
		gt_file_path = os.path.join(gtpath, 'GT_'+ file_id + '.mat')
		gt_list.append(gt_file_path)

	xy = list(zip(image_list, gt_list))		#列表中为元组
	random.shuffle(xy)		   #会更新数据
	s_image_list, s_gt_list = zip(*xy)	#zip(*)可理解为解压

	return s_image_list, s_gt_list

def reshape_tensor(tensor,channel):
	"""
	Reshapes the input tensor appropriate to the network input
	i.e. [1, tensor.shape[0], tensor.shape[1], 1]
	:param tensor: input tensor
	:return: reshaped tensor
	"""
	r_tensor = np.reshape(tensor, newshape=(1, tensor.shape[0], tensor.shape[1], channel))	#
	return r_tensor

def save_weights(graph, fpath):
	"""
	Module to save the weights of the network into a numpy array.
	Saves the weights in .npz file format
	:param graph: Graph whose weights needs to be saved.
	:param fpath: filepath where the weights needs to be saved.
	:return:
	"""
	sess = tf.get_default_session()
	variables = graph.get_collection("variables")
	variable_names = [v.name for v in variables]
	kwargs = dict(zip(variable_names, sess.run(variables)))
	np.savez(fpath, **kwargs)

def load_weights(graph, fpath):
	"""
	Load the weights to the network. Used during transfer learning and for making predictions.
	:param graph: Computation graph on which weights needs to be loaded
	:param fpath: Path where the model weights are stored.
	:return:
	"""
	sess = tf.get_default_session()
	variables = graph.get_collection("variables")
	data = np.load(fpath)
	for v in variables:
		if v.name not in data:
			print("could not load data for variable='%s'" % v.name)
			continue
		print("assigning %s" % v.name)
		sess.run(v.assign(data[v.name]))

def labelmap(img,loc):
	img[loc.astype('int')] = 255
	return img

def get_after_epoch(sess_path,model_final):
	model_list = glob.glob(sess_path+f'/weights.{model_final}.*.npz')
	return model_list

def img_padding(img,shape):#此处的shape采用img.shape得到的形式作为参数,这个函数将拟定于之前reshape多一维前使用
	if len(img.shape)==3:
		img_t = 255.0*np.ones((shape[0],shape[1],3))
	else:img_t = np.zeros((shape[0],shape[1]))#3的是为了彩色图，1的是为了给密度图	 ,1,没办法，下面的img_t要用到第三维度
	raw_shape = img.shape
	raw_h = img.shape[0]
	raw_w = img.shape[1]
	if len(img.shape)==3:
		img_t[:raw_h,:raw_w,:] = img
	else:img_t[:raw_h,:raw_w] = img
	return img_t

def get_size_final(d_map_h,d_map_w):
	d_map_h = 0.5*d_map_h
	if d_map_h!=int(d_map_h):
		d_map_h = int(d_map_h)+1
	d_map_h = 0.5*d_map_h
	if d_map_h!=int(d_map_h):
		d_map_h = int(d_map_h)+1#完成缩小到1/4
	d_map_w = 0.5*d_map_w
	if d_map_w!=int(d_map_w):
		d_map_w = int(d_map_w)+1
	d_map_w = 0.5*d_map_w
	if d_map_w!=int(d_map_w):
		d_map_w = int(d_map_w)+1#完成缩小到1/4
	return int(d_map_h),int(d_map_w)

def rm_weights(path):
	mpath = os.path.split(path)[0]
	filels = os.listdir(mpath)
	filels = [os.path.join(mpath,i) for i in filels if '.npz' in i]
	model_now = os.path.split(path)[-1]
	model_now = model_now.split('.')[1]
	model_now = eval(model_now)
	resls = [os.path.join(mpath,f'weights.{i}.20.npz') for i in range(1,model_now)]
	resls_1 = [os.path.join(mpath,f'weights.{i}.100.npz') for i in range(1,model_now)]
	resls = resls+resls_1
	del resls_1
	resls.append(path)
	resls.append(os.path.join(mpath,f'weights.{model_now}.20.npz'))
	resls_now = [os.path.join(mpath,f'weights.{model_now}.{i}.npz') for i in range(100,model_now,100)]
	resls = resls+resls_now
	del resls_now
	for i in filels:
		if i not in resls:
			os.remove(i)#删除权重文件