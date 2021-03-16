"""
Train script that does full training of the model. It saves the model every epoch.

Before training make sure of the following:

1) The global constants are set i.e. NUM_TRAIN_IMGS, NUM_VAL_IMGS, NUM_TEST_IMGS.
2) The images for training, validation and testing should have proper heirarchy
   and proper file names. Details about the heirarchy and file name convention are
   provided in the README.

Command: python train_model.py --log_dir <log_dir_path> --num_epochs <num_of_epochs> --learning_rate <learning_rate> --session_id <session_id> --data_root <path_of_data>
@author: Aditya Vora
Created on Tuesday Dec 5th, 2017 3:15 PM.
"""

import tensorflow as tf
import src.mccnn as mccnn
import src.layers as L
import os
import src.utils as utils
import numpy as np
import matplotlib.image as mpimg
import scipy.io as sio
import time
import argparse
#import sys
from PIL import Image
import  pylab as plt


# Global Constants. Define the number of images for training, validation and testing.
NUM_TRAIN_IMGS = 6000
NUM_VAL_IMGS = 590
NUM_TEST_IMGS = 587

def main(args):
	"""
	Main function to execute the training.
	Performs training, validation after each epoch and testing after full epoch training.
	:param args: input command line arguments which will set the learning rate, number of epochs, data root etc.
	:return: None
	"""
	
	#args.retrain = True
	
	temp_use_path = os.path.join(args.log_dir, 'session-'+str(args.session_id))
	if os.path.exists(temp_use_path):
		session_id_list = [tempath.split('-')[-1] for tempath in os.listdir(args.log_dir)]
		session_idl = [eval(comid) for comid in session_id_list]
		if session_idl:
			session_id_t = max(session_idl)
		if args.retrain:
			args.session_id = session_id_t
			#args.session_id = 133
		else:
			args.session_id = session_id_t+1
	
	sess_path = utils.create_session(args.log_dir, args.session_id)  # Create a session path based on the session id.
	model_final = 0
	epoch_final = 1
	if args.retrain:
		model_list = os.listdir(sess_path)
		model_list = [i for i in model_list if '.npz' in i]
		model_list = [eval(i.split('.')[2]) for i in model_list]#这样判断出来的是epoch最大
		if model_list:
			model_final = max(model_list)
			model_list = utils.get_after_epoch(sess_path,model_final)
			model_list = [eval(i.split('.')[2]) for i in model_list]

		if model_list:
			epoch_final = max(model_list)
		args.base_model_path = sess_path+f'/weights.{epoch_final}.{model_final}.npz'

	G = tf.Graph()
	with G.as_default():
		# Create image and density map placeholder
		image_place_holder = tf.placeholder(tf.float32, shape=[1, None, None, 3])		 #原本是1，沙雕东西
		d_map_place_holder = tf.placeholder(tf.float32, shape=[1, None, None, 1])

		# Build all nodes of the network
		d_map_est = mccnn.build(image_place_holder)	   #为什么这里要将channel设置为1，不是rgb么，难道不是3通道吗66

		# Define the loss function.
		euc_loss = L.loss(d_map_est, d_map_place_holder)

		# Define the optimization algorithm
		optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)

		# Training node.
		#train_op = optimizer.minimize(euc_loss)

		# Initialize all the variables.
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(graph=G) as sess:
			writer = tf.summary.FileWriter(os.path.join(sess_path,'training_logging'))
			writer.add_graph(sess.graph)
			sess.run(init)							  #完成初始化

			if model_final:
			    utils.load_weights(G, args.base_model_path)

			# Start the epochs
			for eph in range(args.num_epochs):

				start_train_time = time.time()

				# Get the list of train images.
				#train_images_list, train_gts_list = utils.get_data_list(args.data_root, mode='train')
				train_images_list = os.listdir(args.data_root)

				# Loop through all the training images
				for img_idx in range(len(train_images_list)):

					# Load the image and ground truth
					train_image = np.asarray(mpimg.imread(args.data_root+train_images_list[img_idx]), dtype=np.float32)		#卧槽，读图像时直接读成float32啊啊啊啊
					# Reshape the tensor before feeding it to the network
					train_image_r = utils.reshape_tensor(train_image,3)    #将代码修改为根据第二个参数进行设定第4维度
					train_image = np.asarray(mpimg.imread(args.data_root+train_images_list[img_idx]))
					# Prepare feed_dict
					feed_dict_data = {
						image_place_holder: train_image_r
					}

					#输出loss使用的两个输出，查看形状
					if img_idx %1 == 0:
						d_map_view = sess.run(d_map_est,feed_dict = feed_dict_data)
						if args.draw:
							plt.figure()
							plt.imshow(d_map_view[0,:,:,0])
							plt.title(img_idx)
							plt.figure()
							plt.imshow(train_image)
							plt.title(img_idx)
							plt.show()
						print(train_images_list[img_idx])

				end_train_time = time.time()
				train_duration = end_train_time - start_train_time

				# Then we print the results for this epoch:
				print("Epoch {} of {} took {:.3f}s".format(eph + 1, args.num_epochs, train_duration))


if __name__ == "__main__":
	parser = argparse.ArgumentParser() #所以个文件使你只能使用cmd进去输入参数并传参给脚本，并不，有默认参数

	parser.add_argument('--retrain', default=True, type=bool)
	parser.add_argument('--base_model_path', default=None, type=str)
	parser.add_argument('--log_dir', default = './logs', type=str)
	parser.add_argument('--num_epochs', default = 200, type=int)
	parser.add_argument('--learning_rate', default = 0.001, type=float)
	parser.add_argument('--session_id', default = 2, type=int)
	parser.add_argument('--data_root', default='./try_pre/', type=str)
	parser.add_argument('--draw', default = True, type=bool)
	args = parser.parse_args()
	
	while True:
		main(args)
