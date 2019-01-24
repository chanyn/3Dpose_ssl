

import numpy as np
# Make sure that caffe is on the python path:
caffe_root = '../caffe-rpsm/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import cv2
import caffe
import cPickle as pickle

caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../models/mask/p1p2/mask2d3d_iter_650000.caffemodel'
net = caffe.Net('deploy.prototxt', weights, caffe.TEST) 
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

model_save = []
flag = False
for layer_name, param in net.params.iteritems():
    if layer_name == 'change_tmp':
        flag = True
    if flag:
        model_save.append(np.transpose(param[0].data))
        model_save.append(param[1].data)
with open('mask2d3d-320000.pkl','wb') as s:
    pickle.dump(model_save, s)