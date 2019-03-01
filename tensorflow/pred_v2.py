"""Refine 3d poses from difference between predict 2d joints and project 2d joints"""

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
from six.moves import xrange 
import tensorflow as tf
import m_refine_v2
import data
import cPickle as pickle
from skimage import io

tf.app.flags.DEFINE_float("learning_rate", 0.00015, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 30, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")

# Directories
tf.app.flags.DEFINE_string("trained_model", "../models/mask3d-400000", "Training directory.")
tf.app.flags.DEFINE_string("save_root", "results", "save refine result.")

# Data choice
tf.app.flags.DEFINE_string("data_dir", "/data/h36m/", "data root address.")
tf.app.flags.DEFINE_string("coarse3d", "../test/mask3d", "3D coarse prediction.")
tf.app.flags.DEFINE_integer("protocol", 1, "which protocol set will be choosed: 1/3")
tf.app.flags.DEFINE_string("pose2d", "ours", "which 2d prediction will be choosed: ours/hourglass/gt")

FLAGS = tf.app.flags.FLAGS

summaries_dir = 'log'
os.system('mkdir -p {}'.format(summaries_dir))

def create_model(session, batch_size):
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
	pkl_file = FLAGS.trained_model
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

def train(action):
	"""Train a linear model for 3d pose estimation"""
	if FLAGS.protocol == 1:
		test_gt_2d3d, filename = data.readlfile(
			os.path.join(FLAGS.data_dir, '/gt/test/test' + str(action) + '.txt'), 80)
		if FLAGS.pose2d == 'ours':
			test_gt_2d = data.read_2dpredict(
				os.path.join(FLAGS.data_dir, 'ours_2d/bilstm2d-p1-800000/result' + str(action) + '_norm.csv'))
		elif FLAGS.pose2d == 'hourglass':
			print("Use Hourglass Network to predict 2dpose.")
			test_gt_2d = data.readlfile(
				os.path.join(FLAGS.data_dir, 'hg2dh36m/test' + str(action) + '_square2d.txt'), 32)

	elif FLAGS.protocol == 3:
		test_gt_2d3d, filename = data.readlfile(
			os.path.join(FLAGS.data_dir, '/gt/test_p3/test' + str(action) + '.txt'), 80)
		if FLAGS.pose2d == 'ours':
			test_gt_2d = data.read_2dpredict(
				os.path.join(FLAGS.data_dir, 'ours_2d/bilstm2d-p3-800000/result' + str(action) + '_norm.csv'))
		elif FLAGS.pose2d == 'hourglass':
			print("Do not provide Hourglass prediction for Protocol III.")
			exit()
	else:
		print("Unknown Protocol Dataset!")

	pred = data.read_3dpredict(
		os.path.join(FLAGS.coarse3d, 'result' + str(action) + '_norm.csv'), 48, False)

	start_time1 = time.time()
	pred = np.array(pred)
	test_set_2d3d = np.array(test_gt_2d3d)
	if FLAGS.pose2d == 'gt':
		print('Use grpund-turth 2d pose.')
		test_set_2d = test_set_2d3d[:len(pred),:32]
	else:
		test_set_2d = np.array(test_gt_2d)
		test_set_2d = test_set_2d[:len(pred), :]
	test_set_3d = test_set_2d3d[:len(pred), 32:]
	max_min = np.vstack((data.max_16, data.min_16))

	print(pred.shape)
	print(test_set_2d.shape)
	print(test_set_3d.shape)

	print( "done reading and normalizing data." )
	# Avoid using the GPU if requested
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	save_path = os.path.join(FLAGS.save_root, str(FLAGS.learning_rate) + "mpjp_" + str(action) + ".txt")
        save_path1 = os.path.join(FLAGS.save_root, "refine3d_" + str(action) +".txt")

        with tf.Session(config=config) as sess, open(save_path, 'w') as fw, open(save_path1, 'w') as fw1:

	#with tf.Session(config=config) as sess, open(save_path, 'w') as fw:
		FLAGS.batch_size = 1
		print("Creating %d bi-layers of %d units with batchsize %d" % (FLAGS.num_layers, FLAGS.linear_size, FLAGS.batch_size))
		model = create_model(sess, FLAGS.batch_size)
		model.saver.save(sess, "checkpoint/init", global_step=0)
		model.train_writer.add_graph(sess.graph)
		print("Model created")

		encoder_inputs, decoder_2d, decoder_gt3d = model.get_all_batches(pred, test_set_2d, test_set_3d, training=False)
		nbatches = len(encoder_inputs)
		# start_time, loss = time.time(), 0.
		n_joints = 16
		# === Loop through all the training batches ===
		for i in xrange(nbatches):
			print(i)
			enc_in, dec_out, eval3d = encoder_inputs[i], decoder_2d[i], decoder_gt3d[i]
			unnorm_enc_in = enc_in * np.tile((max_min[0] - max_min[1]),(enc_in.shape[0],1)) + np.tile(max_min[1],(enc_in.shape[0],1))
			unnorm_eval3d = eval3d * np.tile((max_min[0] - max_min[1]),(eval3d.shape[0],1)) + np.tile(max_min[1],(eval3d.shape[0],1))

			sqerr = (unnorm_enc_in - unnorm_eval3d)**2 # Squared error between prediction and expected output
			dists = np.zeros((sqerr.shape[0], n_joints)) # Array with L2 error per joint in mm
			dist_idx = 0
			for k in np.arange(0, n_joints*3, 3):
				# Sum across X,Y, and Z dimenstions to obtain L2 distance
				dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
				dist_idx = dist_idx + 1

			print(">>>>>>>> MPJPES loss before refine: {0}\nAverage MPJPES:{1}".format(
				np.sum(dists, axis=1)/n_joints, np.sum(np.sum(dists,axis=1)/n_joints)/dists.shape[0]))
			bf_avg = np.sum(np.sum(dists,axis=1)/n_joints)/dists.shape[0]

			lr = FLAGS.learning_rate
			fw.write("{0}".format(bf_avg))
			start_time = time.time()
			for refine_iter in xrange(2):

				point_loss, bone_loss, enc_in_refine, proj_2d, _ = model.step( sess, dec_out, enc_in, lr)

				unnorm_enc_in_refine = enc_in_refine * np.tile((max_min[0] - max_min[1]), (enc_in_refine.shape[0], 1)) \
									   + np.tile(max_min[1], (enc_in_refine.shape[0], 1))

				sqerr = (unnorm_enc_in_refine - unnorm_eval3d)**2 # Squared error between prediction and expected output
				dists = np.zeros((sqerr.shape[0], n_joints)) # Array with L2 error per joint in mm
				dist_idx = 0
				for k in np.arange(0, n_joints*3, 3):
					# Sum across X,Y, and Z dimenstions to obtain L2 distance
					dists[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k+3], axis=1))
					dist_idx = dist_idx + 1

				mean_mpjp = np.sum(np.sum(dists, axis=1) / n_joints) / dists.shape[0]

				fw.write(" {0}".format(mean_mpjp))
				step_time = (time.time() - start_time)
				print("step_time:{0}ms".format(step_time))

				print("============================\n"
					  "AVERAGE: {0}  Before: {1}\n"
					  "PLoss: {2}    BLoss: {3}\n".format(mean_mpjp, bf_avg, point_loss, bone_loss))

                        fw1.write("{0}\n".format(transwrite(unnorm_enc_in_refine)))
			fw.write("\n")
			model.saver.restore(sess, "checkpoint/init-0")
			sys.stdout.flush()

	step_time = (time.time() - start_time1)
	print("step_time:{0}".format(step_time))
	print("step_time:{0}".format(step_time / nbatches))


def main(_):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	for action in xrange(1, 13):
		if action > 1:
			tf.get_variable_scope().reuse_variables()
		train(action)


if __name__ == "__main__":
	tf.app.run()









