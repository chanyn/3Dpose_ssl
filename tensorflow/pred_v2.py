





"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import copy

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import m_refine_v2
import data
import cPickle as pickle
from skimage import io

tf.app.flags.DEFINE_float("learning_rate", 0.00015, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 30, "How many epochs we should train for")
# tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
# tf.app.flags.DEFINE_boolean("max_norm", False, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")

# Data loading
# tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
# tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")

# Evaluation
# tf.app.flags.DEFINE_boolean("evaluateActionWise",False, "The dataset to use either h36m or heva")

# Directories
tf.app.flags.DEFINE_string("train_dir", "../tools/mask2d3d-", "Training directory.")

# Train or load
tf.app.flags.DEFINE_integer("load", 650000, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_string("save_root", "results", "save refine result.")



FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir

print( train_dir )
summaries_dir = 'log'
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, batch_size):
	"""
	Create model and initialize it or load its parameters in a session

	Args
		session: tensorflow session
		batch_size: integer. Number of examples in each batch
	Returns
		model: The created (or loaded) model
	Raises
		ValueError if asked to load a model, but the checkpoint specified by
		FLAGS.load cannot be found.
	"""
	pkl_file = train_dir + str(FLAGS.load) + '.pkl'
	print("load from {0}".format(pkl_file))
	model = m_refine_v2.LinearModel(
			FLAGS.linear_size,
			FLAGS.num_layers,
			FLAGS.residual,
			FLAGS.batch_norm,
			batch_size,
			FLAGS.learning_rate,
			summaries_dir,
			pkl_file,
			dtype=tf.float32)

	# initial
	session.run( tf.global_variables_initializer())

	return model

def transwrite(array):
	array = np.squeeze(array)
	norm_labels = array.tolist()
	norm_labels = [str(x) for x in norm_labels]
	label_str = ','.join(norm_labels)
	return label_str

def train(action,phase):
	"""Train a linear model for 3d pose estimation"""

	test_gt_2d3d, filename = data.readlfile('/home/cyan/data/human3.6m/annotation/16test/test' + str(action) + '.txt', 80)
	# test_gt_3d, filename = data.readlfile('/home/cyan/data/human3.6m/annotation/test_p3/16test' + str(action) + '.txt', 48)
	test_gt_2d = data.read_2dpredict('/home/cyan/code/caffe-rpsm/examples/2d/test/bilstm2d-p1-800000/result'+str(action)+'_norm.csv')
	pred = data.read_3dpredict('/home/cyan/cp_to_shenji/12/h36m-p1/mask3d-400000/result'+str(action)+'_norm.csv',48,False)
	
	start_time1 = time.time()
	test_set_2d3d = np.array(test_gt_2d3d)
	# test_set_2d = test_set_2d3d[:,:32]
	test_set_2d = np.array(test_gt_2d)
	test_set_3d = test_set_2d3d[:,32:]
	# test_set_3d = np.array(test_gt_3d)
	max_min = np.vstack((data.max_16, data.min_16))
	# sys.exit(0)

	pred = np.array(pred)

	test_set_2d = test_set_2d[:len(pred),:]
	test_set_3d = test_set_3d[:len(pred),:]

	print(pred.shape)
	print(test_set_2d.shape)
	print(test_set_3d.shape)

	print( "done reading and normalizing data." )
	# Avoid using the GPU if requested
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	save_path = os.path.join(FLAGS.save_root,str(FLAGS.learning_rate) + "mpjp_" + str(action) +".txt")
	# save_path1 = os.path.join(FLAGS.save_root, "mask3d_record_" + str(action) +"_s1.txt")
	# save_path2 = os.path.join(FLAGS.save_root, "mask3d_record_" + str(action) +"_s2.txt")
	# save_path3 = os.path.join(FLAGS.save_root, "mask3d_record_" + str(action) +"_s3.txt")
	# save_path4 = os.path.join(FLAGS.save_root, "mask3d_record_" + str(action) +"_s4.txt")
	# save_path5 = os.path.join(FLAGS.save_root, "mask3d_record_" + str(action) +"_s5.txt")
	# # === Create the model ===
	# with tf.Session(config=config) as sess, open(save_path, 'w') as fw, open(save_path1, 'w') as fw1, open(save_path2, 'w') as fw2,open(save_path3, 'w') as fw3, open(save_path4, 'w') as fw4,open(save_path5, 'w') as fw5:
	# 	dic = {0:fw1, 1:fw2, 2:fw3, 3:fw4, 4:fw5}
	with tf.Session(config=config) as sess, open(save_path, 'w') as fw:
		FLAGS.batch_size = 1
		print("Creating %d bi-layers of %d units with batchsize %d" % (FLAGS.num_layers, FLAGS.linear_size, FLAGS.batch_size))
		model = create_model( sess, FLAGS.batch_size)
		model.saver.save(sess,"checkpoint/init",global_step=0) 
		model.train_writer.add_graph( sess.graph )
		print("Model created")

		encoder_inputs, decoder_2d, decoder_gt3d = model.get_all_batches( pred, test_set_2d, test_set_3d, training=False )
		nbatches = len( encoder_inputs )
		# start_time, loss = time.time(), 0.
		n_joints = 16
		# === Loop through all the training batches ===
		for i in xrange( nbatches ):
			print(i)
			# img=io.imread(filename[i])
			# bbox = max(img.shape)
			enc_in, dec_out, eval3d = encoder_inputs[i], decoder_2d[i], decoder_gt3d[i]
			unnorm_enc_in = enc_in * np.tile((max_min[0] - max_min[1]),(enc_in.shape[0],1)) + np.tile(max_min[1],(enc_in.shape[0],1))
			unnorm_eval3d = eval3d * np.tile((max_min[0] - max_min[1]),(eval3d.shape[0],1)) + np.tile(max_min[1],(eval3d.shape[0],1))
			# unnorm_eval3d = eval3d

			sqerr = (unnorm_enc_in - unnorm_eval3d)**2 # Squared error between prediction and expected output
			dists = np.zeros( (sqerr.shape[0], n_joints ) ) # Array with L2 error per joint in mm
			dist_idx = 0
			for k in np.arange(0, n_joints*3, 3):
				# Sum across X,Y, and Z dimenstions to obtain L2 distance
				dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
				dist_idx = dist_idx + 1

			print(">>>>>>>> MPJPES loss before refine: {0}\nAverage MPJPES:{1}".format(np.sum(dists,axis=1)/n_joints, np.sum(np.sum(dists,axis=1)/n_joints)/dists.shape[0]))
			bf_avg = np.sum(np.sum(dists,axis=1)/n_joints)/dists.shape[0]

			# sess.run(tf.variables_initializer(tf.trainable_variables()))
			# model.saver.restore(sess, "checkpoint/init-0")

			lr = FLAGS.learning_rate
			fw.write("{0}".format(bf_avg))
			start_time = time.time()
			for refine_iter in xrange(2):

				point_loss, bone_loss, enc_in_refine, proj_2d, _ =  model.step( sess, dec_out, enc_in, lr)

				unnorm_enc_in_refine = enc_in_refine * np.tile((max_min[0] - max_min[1]),(enc_in_refine.shape[0],1)) + np.tile(max_min[1],(enc_in_refine.shape[0],1))
				# unnorm_proj_2d = proj_2d * np.tile(bbox,(proj_2d.shape[0],proj_2d.shape[1])) 

				sqerr = (unnorm_enc_in_refine - unnorm_eval3d)**2 # Squared error between prediction and expected output
				dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
				dist_idx = 0
				for k in np.arange(0, n_joints*3, 3):
					# Sum across X,Y, and Z dimenstions to obtain L2 distance
					dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
					dist_idx = dist_idx + 1

				mean_mpjp = np.sum(np.sum(dists,axis=1)/n_joints)/dists.shape[0]

				# dic[refine_iter].write("{0}\n".format(transwrite(unnorm_enc_in_refine)))
                # dic[refine_iter].write("{0}\n".format(transwrite(unnorm_enc_in_refine),transwrite(unnorm_proj_2d)))

				fw.write(" {0}".format(mean_mpjp))
				step_time = (time.time() - start_time)
				print("step_time:{0}ms".format(step_time))

				print("============================\n"
				 	  "AVERAGE: {0}  Before: {1}\n"
				 	  "PLoss: {2}    BLoss: {3}\n".format(mean_mpjp, bf_avg, point_loss, bone_loss))			

			fw.write("\n")
			model.saver.restore(sess, "checkpoint/init-0")

		# Reset global time and loss
			step_time = 0
			sys.stdout.flush()
	step_time = (time.time() - start_time1)
	print("step_time:{0}".format(step_time))
	print("step_time:{0}".format(step_time / nbatches))


def main(_):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	phase = 'test'
	# for action in xrange(1,13):
	# 	if action > 1:
	# 		tf.get_variable_scope().reuse_variables()
	# 	train(action, phase)
	train(1, phase)


if __name__ == "__main__":
	tf.app.run()









