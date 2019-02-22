import numpy as np
import sys
import cv2
import caffe
import cPickle as pickle

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('caffe_root', help='test config file path')
    parser.add_argument('caffemodel', help='checkpoint file')
    parser.add_argument('--prototxt', help='Network Prototxt', default='deploy.prototxt', type=str)
    parser.add_argument('--pkl_dir', help='output pkl file', default='model.pkl', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Make sure that caffe is on the python path:
    sys.path.insert(0, args.caffe_root + 'python')
    caffe.set_device(0)
    caffe.set_mode_gpu()

    weights = args.caffemodel
    net = caffe.Net(args.prototxt, weights, caffe.TEST)
    for layer_name, param in net.params.iteritems():
        print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

    model_save = []
    flag = False
    for layer_name, param in net.params.iteritems():
        if layer_name == 'change_tmp':
            flag = True
        if flag:
            model_save.append(np.transpose(param[0].data))
            model_save.append(param[1].data)
    with open(args.pkl_dir, 'wb') as s:
        pickle.dump(model_save, s)

if __name__ == '__main__':
    main()