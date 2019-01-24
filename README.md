## 3D Human Pose Machines with Self-supervised Learning

Keze Wang, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei, [“3D Human Pose Machines with Self-supervised Learning”](https://arxiv.org/abs/1901.03798). To appear in IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2019.

This repository implements a 3D human pose machine to resolve 3D pose sequence generation for monocular frames, and includes a concise self-supervised correction mechanism to enhance our model by retaining the 3D geometric consistency. A mainly part is written in C++ and powered by [Caffe](https://github.com/BVLC/caffe) deep learning toolbox. Another is written in Python and powered by [Tensorflow]().

### Results

We proposed results on the Human3.6M, KTH Football II and MPII dataset.

<p align="center">
    <img src=http://www.sysu-hcp.net/wp-content/uploads/2019/01/WeChat-Screenshot_20190111210435.png>
</p>

<p align="center">
    <img src="http://www.sysu-hcp.net/wp-content/uploads/2019/01/WeChat-Screenshot_20190111210519.png">
</p>

<p align="center">
    <img src="http://www.sysu-hcp.net/wp-content/uploads/2019/01/WeChat-Screenshot_20190111210546.png">
</p>

### Citation



### Get Started

Clone the repo:

```
git clone https://github.com/chanyn/3Dpose_ssl.git
```

Our code is orgamized as follows:

```
caffe-3dssl/: support caffe
models/: pretrained models and results
prototxt/: network architecture definitions
tensorflow/: code for online refine 
test/: script that run results split by action 
tools/: python and matlab code 
```

#### Requirements

1. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0 has been tested. The other versions should be working.
2. Caffe Python wrapper is required. 
3. Tensorflow 
4. python 2.7
5. MATLAB
6. Opencv-python

#### Installation

1. Build 3Dssl Caffe

   ```
   cd $ROOT/caffe-3dssl
   # Follow the Caffe installation instructions here:
   #   http://caffe.berkeleyvision.org/installation.html
   
   # If you're experienced with Caffe and have all of the requirements installed
   # and your Makefile.config in place, then simply do:
   make all -j 8
   
   make pycaffe
   ```

2. Install Tensorflow

#### Datasets

We have provided [protocol #I and protocol #III]() split list of Human3.6m. Follow [Human3.6m website](http://vision.imar.ro/human3.6m/description.php) to download videos and API. We split each video per 5 frames.  

```
h36m
|_gt
|_hg2dh36m
|_ours_2d
|_ours_3d
|_16skel_train_2d3d_clip.txt
|_16skel_test_2d3d_clip.txt
|_16skel_train_2d3d_p3_clip.txt
|_16skel_test_2d3d_p3_clip.txt
```

After set up Human3.6m dataset following its illustration and download above training/testing list. You should update paths in [***prototxt***]() for images and annotation director.

###Training

Our framework training is consist of offline pharse and online pharse.

We provide pretrained [CPN-caffemodel](). Please put it into *models/*.

#### Offline

```
# offline training
# you can change initial weights or prototxt
sh caffe_3dssl/examples/train.sh 
```

#### Online SSL

```
# save 2d or 3d coarse prediction
sh test_all.sh

# transfer caffemodel of SSL module to tensorflow type
python tools/gen_refinepkl.py

# online training
# record final prediction
python tensorflow/pred_v2.py
```

#### Evaluation

```
run tools/eval_h36m.m
```



