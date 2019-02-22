"""Simple model to regress 3d human poses from 2d joint locations"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cPickle as pickle
import copy


def kaiming(shape, dtype, partition_info=None):
  """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

  Args
    shape: dimensions of the tf array to initialize
    dtype: data type of the array
    partition_info: (Optional) info about how the variable is partitioned.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
      Needed to be used as an initializer.
  Returns
    Tensorflow array with initial weights
  """
  return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class LinearModel(object):
  """ A simple Linear+RELU model """

  def __init__(self,
               linear_size,
               num_layers,
               residual,
               batch_norm,
               batch_size,
               learning_rate,
               summaries_dir,
               param_path,
               dtype=tf.float32):
    """Creates the linear + relu model

    Args
      linear_size: integer. number of units in each layer of the model
      num_layers: integer. number of bilinear blocks in the model
      residual: boolean. Whether to add residual connections
      batch_norm: boolean. Whether to use batch normalization
      batch_size: integer. The size of the batches used during training
      learning_rate: float. Learning rate to start with
      summaries_dir: String. Directory where to log progress
      predict_14: boolean. Whether to predict 14 instead of 17 joints
      dtype: the data type to use to store internal variables
    """

    # There are in total 17 joints in H3.6M and 16 in MPII (and therefore in stacked
    # hourglass detections). We settled with 16 joints in 2d just to make models
    # compatible (e.g. you can train on ground truth 2d and test on SH detections).
    self.HUMAN_2D_SIZE = 16 * 2

    # In 3d all the predictions are zero-centered around the root (hip) joint, so
    # we actually predict only 16 joints.
    self.HUMAN_3D_SIZE = 16 * 3

    self.input_size  = self.HUMAN_3D_SIZE
    self.output_size = self.HUMAN_2D_SIZE

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter( os.path.join(summaries_dir, 'train'))
    self.test_writer  = tf.summary.FileWriter( os.path.join(summaries_dir, 'test'))
    self.batch_norm = batch_norm
    self.num_layers = num_layers
    self.residual = residual

    self.linear_size   = linear_size
    self.batch_size    = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
    self.global_step   = tf.Variable(0, trainable=False, name="global_step")
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    self.mask = tf.constant([1.,1.,1.,1.,1.,1.,1.,1. ,1.,1.,1.,1.,1.,1. ,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1. ,1.,1.,1.,1.,1.,1.])
    self.isTraining = False
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    self.dec_out = tf.placeholder(dtype, shape=[None, self.output_size], name="dec_out")
    self.enc_in = tf.placeholder(dtype, shape=[self.batch_size, self.input_size], name="enc_in")
      
    pkl_file = open(param_path, "rb")
    self.pretrain_param = pickle.load(pkl_file) 

    # === Create the linear + relu combos ===
    with vs.variable_scope("linear_model" ):
      w0 = tf.get_variable(name="w0", initializer=tf.constant_initializer(self.pretrain_param[0]),
                           shape=[self.HUMAN_3D_SIZE, linear_size] , dtype=dtype,trainable=True)
      b0 = tf.get_variable(name="b0", initializer=tf.constant_initializer(self.pretrain_param[1]),
                           shape=[linear_size], dtype=dtype ,trainable=True)
      y0 = tf.matmul( self.enc_in, w0 ) + b0
      y0 = tf.nn.relu( y0 )
      w00 = tf.get_variable(name="w00", initializer=tf.constant_initializer(self.pretrain_param[2]),
                            shape=[linear_size, self.HUMAN_3D_SIZE], dtype=dtype,trainable=True)
      b00 = tf.get_variable(name="b00", initializer=tf.constant_initializer(self.pretrain_param[3]),
                            shape=[self.HUMAN_3D_SIZE], dtype=dtype ,trainable=True)
      y00 = tf.matmul( y0, w00 ) + b00

      # === First layer, brings dimensionality up to linear_size ===
      w1 = tf.get_variable(name="w1", initializer=tf.constant_initializer(self.pretrain_param[4]),
                           shape=[self.HUMAN_3D_SIZE, self.linear_size], dtype=dtype, trainable=False )
      b1 = tf.get_variable(name="b1", initializer=tf.constant_initializer(self.pretrain_param[5]),
                           shape=[self.linear_size], dtype=dtype ,trainable=False)
      y1 = tf.matmul( y00, w1 ) + b1
      if self.batch_norm:
        y1 = tf.layers.batch_normalization(y1,training=self.isTraining, name="batch_normalization",trainable=False)
      y1 = tf.nn.relu( y1 )
      #y1 = tf.nn.dropout( y1, self.dropout_keep_prob )

      # === Create multiple bi-linear layers ===
      y2 = self.two_linear(y1, self.linear_size, self.residual, self.dropout_keep_prob, self.batch_norm, dtype, 6 )
      
      w4 = tf.get_variable(name="w4", initializer=tf.constant_initializer(self.pretrain_param[10]),
                           shape=[self.linear_size, self.HUMAN_2D_SIZE], dtype=dtype,trainable=False)
      b4 = tf.get_variable(name="b4", initializer=tf.constant_initializer(self.pretrain_param[11]),
                           shape=[self.HUMAN_2D_SIZE], dtype=dtype,trainable=False)
      y = tf.matmul(y2, w4) + b4
      # === End linear model ===

    # Store the outputs here
    self.W = w0
    self.refine = y00
    self.proj = y

    mask_y = y * self.mask
    mask_gt = self.dec_out * self.mask
    self.bone_loss = tf.reduce_mean(tf.square(self.compute_bone(mask_y,16) - self.compute_bone(mask_gt,16)))
    self.point_loss = tf.reduce_mean(tf.square(mask_y  - mask_gt))

    self.loss = self.bone_loss + self.point_loss
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    # Gradients and update operation for training the model.
    self.updates = tf.train.AdamOptimizer( self.learning_rate).minimize(self.loss)
    self.saver = tf.train.Saver([w0]+[b0]+[w00]+[b00])

  def two_linear(self, xin, linear_size, residual, dropout_keep_prob, batch_norm, dtype, idx ):
    """
    Make a bi-linear block with optional residual connection

    Args
      xin: the batch that enters the block
      linear_size: integer. The size of the linear units
      residual: boolean. Whether to add a residual connection
      dropout_keep_prob: float [0,1]. Probability of dropping something out
      batch_norm: boolean. Whether to do batch normalization
      dtype: type of the weigths. Usually tf.float32
      idx: integer. Number of layer (for naming/scoping)
    Returns
      y: the batch after it leaves the block
    """
    with vs.variable_scope("two_linear_"+str(idx)) as scope:

      input_size = int(xin.get_shape()[1])

      # Linear 1
      w2 = tf.get_variable(name="w2_"+str(idx), initializer=tf.constant_initializer(self.pretrain_param[idx]),
                            shape=[input_size, linear_size], dtype=dtype,trainable=False)
      b2 = tf.get_variable(name="b2_"+str(idx), initializer=tf.constant_initializer(self.pretrain_param[idx+1]),
                            shape=[linear_size], dtype=dtype,trainable=False)
      y = tf.matmul(xin, w2) + b2
      if batch_norm:
        y = tf.layers.batch_normalization(y, training=self.isTraining,name="batch_normalization1"+str(idx),trainable=False)
      y = tf.nn.relu( y )
      #y = tf.nn.dropout( y, dropout_keep_prob )

      # Linear 2
      w3 = tf.get_variable(name="w3_"+str(idx), initializer=tf.constant_initializer(self.pretrain_param[idx+2]),
                           shape=[linear_size, linear_size], dtype=dtype,trainable=False)
      b3 = tf.get_variable(name="b3_"+str(idx), initializer=tf.constant_initializer(self.pretrain_param[idx+3]),
                           shape=[linear_size], dtype=dtype,trainable=False)
      y = tf.matmul(y, w3) + b3
      if batch_norm:
        y = tf.layers.batch_normalization(y,training=self.isTraining,name="batch_normalization2"+str(idx),trainable=False)
      y = tf.nn.relu( y )
      # y = tf.nn.dropout( y, dropout_keep_prob)

      # Residual every 2 blocks
      y = (xin + y) if residual else y

    return y

  def compute_bone(self,x,skel_num):
      parent_node = [0, 0,1,2, 0,4,5, 0,7,8, 8,10,11, 8,13,14]
      x_1 = []
      for i in xrange(skel_num):
        x_1.append(x[:,2*parent_node[i]])
        x_1.append(x[:,2*parent_node[i]+1])
      x_p = tf.transpose(tf.stack(x_1,axis=0))
      diff = tf.abs(x_p - x)
      return diff


  def step_loss(self, session, decoder_outputs, eval3d):
    input_feed = {self.dec_out: decoder_outputs, self.enc_in: eval3d}

    # Output feed: depends on whether we do a backward step or not.
    output_feed = self.loss
    outputs = session.run( output_feed, input_feed )

    return outputs

  def step(self, session, decoder_outputs, eval3d, lr):
    input_feed = {self.dec_out: decoder_outputs, self.enc_in: eval3d, self.learning_rate: lr}

    # Output feed: depends on whether we do a backward step or not.
    output_feed = [self.updates,       # Update Op that does SGD
                   self.point_loss,
                   self.bone_loss,
                   self.refine,
                   self.proj,
                   self.W]

    outputs = session.run( output_feed, input_feed )
    return outputs[1], outputs[2], outputs[3], outputs[4] ,outputs[5]

  def get_all_batches( self, data_x, data_y, gt_3d, training=True ):
    assert data_x.shape[0] == data_y.shape[0]
    assert data_x.shape[0] == gt_3d.shape[0]
    encoder_inputs  = data_x
    decoder_outputs = data_y
    n = encoder_inputs.shape[0]
    gt_3d_out = gt_3d

    if training:
      # Randomly permute everything
      idx = np.random.permutation( n )
      encoder_inputs  = encoder_inputs[idx, :]
      decoder_outputs = decoder_outputs[idx, :]
      gt_3d_out = gt_3d[idx,:]

    # Make the number of examples a multiple of the batch size
    n_extra  = n % self.batch_size
    if n_extra > 0:  # Otherwise examples are already a multiple of batch size
      encoder_inputs  = encoder_inputs[:-n_extra, :]
      decoder_outputs = decoder_outputs[:-n_extra, :]
      gt_3d_out = gt_3d[:-n_extra,:]

    n_batches = n // self.batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )
    gt_3d_out = np.split( gt_3d_out, n_batches )

    return encoder_inputs, decoder_outputs, gt_3d_out

