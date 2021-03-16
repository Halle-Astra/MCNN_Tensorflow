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
#from PIL import Image
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
	bsize = args.bsize
	print(bsize)
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
		model_list = [eval(i.split('.')[1]) for i in model_list]#这样判断出来的是epoch最大  ,感觉放以前，这应该model才对吧
		if model_list:
			model_final = max(model_list)
			model_list = utils.get_after_epoch(sess_path,model_final)
			model_list = [eval(i.split('.')[-2]) for i in model_list]

		if model_list:
			epoch_final = max(model_list)
			#model_final = max(model_list)
		#args.base_model_path = sess_path+f'/weights.{epoch_final}.{model_final}.npz'
		args.base_model_path = sess_path+f'/weights.{model_final}.{epoch_final}.npz'

	G = tf.Graph()
	with G.as_default():
		# Create image and density map placeholder
		image_place_holder = tf.placeholder(tf.float32, shape=[bsize, None, None, 3])		 #原本是1，沙雕东西
		d_map_place_holder = tf.placeholder(tf.float32, shape=[bsize, None, None, 1])
		#shape_place_holder = tf.placeholder(tf.float32,shape=[bsize,2])#可以采用这种形式给，之后可以由列表赋值
		hide_place_holder = tf.placeholder(tf.float32,shape = [bsize,None,None,1])#用于遮罩

		# Build all nodes of the network
		d_map_est = mccnn.build(image_place_holder)	   #为什么这里要将channel设置为1，不是rgb么，难道不是3通道吗66

		# Define the loss function.
		euc_loss = L.loss(d_map_est, d_map_place_holder,hide_place_holder)

		# Define the optimization algorithm
		optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)

		# Training node.
		train_op = optimizer.minimize(euc_loss)

		# Initialize all the variables.
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		# For summary
		summary = tf.summary.merge_all()

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
				train_images_list, train_gts_list = utils.get_data_list(args.data_root, mode='train')
				total_train_loss = 0

				img_ls_train = []
				d_map_ls_train = []
				# Loop through all the training images
				for img_idx in range(len(train_images_list)):

					# Load the image and ground truth
					train_image = np.asarray(mpimg.imread(train_images_list[img_idx]), dtype=np.float32)		#卧槽，读图像时直接读成float32啊啊啊啊
					train_d_map = np.asarray(sio.loadmat(train_gts_list[img_idx])['image_info'][0][0][0][0][0], dtype=np.float32)	  #为了得到个数据，我是没见过比这更有病的

					# Reshape the tensor before feeding it to the network
					#train_image_r = utils.reshape_tensor(train_image,3)    #将代码修改为根据第二个参数进行设定第4维度	,好吧，这一步留给凑给10个后再做

					#将train_d_map中的坐标信息转换成图像信息
					#d_map_from_loc = np.zeros(train_image.shape[:2],dtype = 'float32')#from location，先制造出一张图像，考虑将图片制作成灰度图,优先考虑使用uint8的255做一个人的位置
					#for i in range(int(train_d_map.size/2+1)):
						#d_map_from_loc = utils.labelmap(d_map_from_loc,i)
					#完成对图像人头位置数据的图像化,之后应该是完成对图像的高斯卷积（获得密度图）


					train_d_map_r = utils.get_density_map_gaussian(train_d_map,train_image.shape[0],train_image.shape[1]) #通过高斯卷积（模糊）得到密度图
					
					'''以下的resize操作已不需要了，对于图像相加求和等于人数而言，这一步也是有害的。但reshape多一维的操作还是要的。之后凑齐十个一起干。
					d_map_rs_h,d_map_rs_w = train_d_map_r.shape
					train_d_map_r = Image.fromarray(train_d_map_r)
					train_d_map_r = train_d_map_r.resize((int(d_map_rs_w*mccnn.csz),int(d_map_rs_h*mccnn.csz)))
					train_d_map_r = np.asarray(train_d_map_r)
					train_d_map_r = utils.reshape_tensor(train_d_map_r,1)
					'''
					
					img_ls_train.append(train_image)
					d_map_ls_train.append(train_d_map_r)
					if (img_idx+1)%bsize!=0:	 #后面的都是训练，等到了10个再走后面的流程
						continue 
					max_h = max([i.shape[0] for i in img_ls_train])			   
					max_w = max([i.shape[1] for i in img_ls_train])#拿到两个的最大值，然后准备填充
					max_h_final,max_w_final = utils.get_size_final(max_h,max_w)
					shape_ls_train = [[i.shape[0],i.shape[1]] for i in img_ls_train]
					img_ls_train = np.concatenate([utils.reshape_tensor(utils.img_padding(i,(max_h,max_w)),3) for i in img_ls_train],axis = 0)
					d_map_ls_train = np.concatenate([utils.reshape_tensor(utils.img_padding(i,(max_h_final,max_w_final)),1) for i in d_map_ls_train],axis = 0)
					hide_ls = np.zeros(d_map_ls_train.shape,dtype = 'float32')
					for i in range(bsize):
						hide_ls[i,:shape_ls_train[i][0],:shape_ls_train[i][1],:] = 1
				

					# Prepare feed_dict
					feed_dict_data = {
						image_place_holder: img_ls_train,
						d_map_place_holder: d_map_ls_train,
						hide_place_holder: hide_ls
					}
					
					#查看d_map_est的shape
					#estview = sess.run(d_map_est,feed_dict = feed_dict_data)

					# Compute the loss for one image.
					sess.run(train_op, feed_dict=feed_dict_data)

					#输出loss使用的两个输出，查看形状
					if (img_idx+1) %300 == 0:
						d_map_view,d_map_true,loss_per_image = sess.run([d_map_est,d_map_place_holder,euc_loss],feed_dict = feed_dict_data)
						if args.draw:
							plt.figure()
							plt.imshow(d_map_view[0,:,:,0])
							plt.title(img_idx)
							plt.figure()
							plt.imshow(d_map_true[0,:,:,0])
							plt.title(img_idx)
							plt.show()
						
						print(train_images_list[img_idx])
						print(f'Loss of {img_idx} is :{loss_per_image}')					#懒得修改为loss_per_batch了
						
						utils.save_weights(G, os.path.join(sess_path, "weights.%s.%s" % (model_final+1,eph+1)))
						utils.rm_weights(os.path.join(sess_path, "weights.%s.%s.npz" % (model_final+1,eph+1)))
						summary_str = sess.run(summary, feed_dict=feed_dict_data)
						writer.add_summary(summary_str, eph)
					# Accumalate the loss over all the training images.		#accumalate，累计的
						total_train_loss = total_train_loss + loss_per_image   #total仅仅用于输出，无法利于优化
					
					img_ls_train = []
					d_map_ls_train = []
					hide_ls = []

				end_train_time = time.time()
				train_duration = end_train_time - start_train_time

				# Compute the average training loss
				avg_train_loss = total_train_loss / len(train_images_list)

				# Then we print the results for this epoch:
				print("Epoch {} of {} took {:.3f}s".format(eph + 1, args.num_epochs, train_duration))
				print("  Training loss:\t\t{:.6f}".format(avg_train_loss))

'''
				print ('Validating the model...')

				total_val_loss = 0

				# Get the list of images and the ground truth
				val_image_list, val_gt_list = utils.get_data_list(args.data_root, mode='valid')

				valid_start_time = time.time()

				# Loop through all the images.
				for img_idx in range(len(val_image_list)):

					# Read the image and the ground truth
					val_image = np.asarray(mpimg.imread(val_image_list[img_idx]), dtype=np.float32)
					val_d_map = np.asarray(sio.loadmat(val_gt_list[img_idx])['d_map'], dtype=np.float32)

					# Reshape the tensor for feeding it to the network
					val_image_r = utils.reshape_tensor(val_image)
					val_d_map_r = utils.get_density_map_gaussian(val_d_map,val_image.shape[0],val_image.shape[1])
					
					d_map_rs_h,d_map_rs_w = val_d_map_r.shape
					val_d_map_r = Image.fromarray(val_d_map_r)
					val_d_map_r = val_d_map_r.resize(int(d_map_rs_w*mccnn.csz),int(d_map_rs_h*mccnn.csz))
					val_d_map_r = np.asarray(val_d_map_r)
					val_d_map_r = utils.reshape_tensor(val_d_map_r,1)

					# Prepare the feed_dict
					feed_dict_data = {
						image_place_holder: val_image_r,
						d_map_place_holder: val_d_map_r,
					}

					# Compute the loss per image
					loss_per_image = sess.run(euc_loss, feed_dict=feed_dict_data)

					# Accumalate the validation loss across all the images.
					total_val_loss = total_val_loss + loss_per_image

				valid_end_time = time.time()
				val_duration = valid_end_time - valid_start_time

				# Compute the average validation loss.
				avg_val_loss = total_val_loss / len(val_image_list)

				print("  Validation loss:\t\t{:.6f}".format(avg_val_loss))
				print ("Validation over {} images took {:.3f}s".format(len(val_image_list), val_duration))

				# Save the weights as well as the summary
				utils.save_weights(G, os.path.join(sess_path, "weights.%s" % (eph+1)))
				summary_str = sess.run(summary, feed_dict=feed_dict_data)
				writer.add_summary(summary_str, eph)


			print ('Testing the model with test data.....')

			# Get the image list
			test_image_list, test_gt_list = utils.get_data_list(args.data_root, mode='test')
			abs_err = 0

			# Loop through all the images.
			for img_idx in range(len(test_image_list)):

				# Read the images and the ground truth
				test_image = np.asarray(mpimg.imread(test_image_list[img_idx]), dtype=np.float32)
				test_d_map = np.asarray(sio.loadmat(test_gt_list[img_idx])['d_map'], dtype=np.float32)                

				# Reshape the input image for feeding it to the network.
				test_image = utils.reshape_tensor(test_image)
				feed_dict_data = {image_place_holder: test_image}

				# Make prediction.
				pred = sess.run(d_map_est, feed_dict=feed_dict_data)                

				# Compute mean absolute error.
				abs_err += utils.compute_abs_err(pred, test_d_map)

			# Average across all the images.
			avg_mae = abs_err / len(test_image_list)
			print ("Mean Absolute Error over the Test Set: %s" %(avg_mae))
			print ('Finished.')
'''

if __name__ == "__main__":
	parser = argparse.ArgumentParser() #所以个文件使你只能使用cmd进去输入参数并传参给脚本，并不，有默认参数

	parser.add_argument('--retrain', default=True, type=bool)
	parser.add_argument('--base_model_path', default=None, type=str)
	parser.add_argument('--log_dir', default = './logs', type=str)
	parser.add_argument('--num_epochs', default = 200, type=int)
	parser.add_argument('--bsize', default = 10, type=int)
	parser.add_argument('--learning_rate', default = 0.001, type=float)
	parser.add_argument('--session_id', default = 2, type=int)
	parser.add_argument('--data_root', default='./data/comb_dataset_v3', type=str)
	parser.add_argument('--draw', default = False, type=bool)
	args = parser.parse_args()
	
	#if args.retrain:
	#    if args.base_model_path is None:
	#        print "Please provide a base model path."
	#        sys.exit()
	#    else:
	#        main(args)
	#else:
	while True:
		main(args)
