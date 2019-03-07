## 3D Human Pose Machines with Self-supervised Learning

Keze Wang, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei, [“3D Human Pose Machines with Self-supervised Learning”](https://arxiv.org/abs/1901.03798). To appear in IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2019.

This repository implements a 3D human pose machine to resolve 3D pose sequence generation for monocular frames, and includes a concise self-supervised correction mechanism to enhance our model by retaining the 3D geometric consistency. A mainly part is written in C++ and powered by [Caffe](https://github.com/BVLC/caffe) deep learning toolbox. Another is written in Python and powered by [Tensorflow](https://github.com/tensorflow/tensorflow).

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

### License

This project is released for Adamic Research Use only.

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
3. Tensorflow 1.1.0
4. python 2.7.13
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

+ Human3.6m

  We change annotation of Human3.6m to hold 16 points ( 'RFoot' 'RKnee' 'RHip' 'LHip' 'LKnee' 'LFoot' 'Hip' 'Spine' 'Thorax' 'Head' 'RWrist' 'RElbow'  'RShoulder' 'LShoulder' 'LElbow' 'LWrist') in keeping with MPII. 

  We have provided [count mean file and protocol #I & protocol #III split list of Human3.6m.](https://drive.google.com/open?id=1uXy_BdfS8nt6dghSI6gDqWmcB84_jl3o) Follow [Human3.6m website](http://vision.imar.ro/human3.6m/description.php) to download videos and API. We split each video per 5 frames, you can directly download processed square data in this [link](https://pan.baidu.com/s/1ieLHH9w8tnKPcB836Jxtcw).  And list format of 16skel_train/test_* is [img_path] [P1<sub>2dx</sub>, P1<sub>2dy</sub>, P2<sub>2dx</sub>, P2<sub>2dy</sub>,..., P1<sub>3dx</sub>, P1<sub>3dy</sub>, P1<sub>3dz</sub>, P2<sub>3dx</sub>, P2<sub>3dy</sub>, P2<sub>3dz</sub>,...] clip. Clip = 0 denote reset lstm.

  ```shell
  # files construction
  h36m
  |_gt # 2d and 3d annotations splited by actions
  |_hg2dh36m # 2d estimation predicted by *Hourglass*, 'square' denotes prediction of square image. 
  |_ours_2d # 2d prediction from our model
  |_ours_3d # 3d coarse prediction of *Model Extension: mask3d*
  |_16skel_train_2d3d_clip.txt # train list of *Protocol I*
  |_16skel_test_2d3d_clip.txt
  |_16skel_train_2d3d_p3_clip.txt # train list of *Protocol III*
  |_16skel_test_2d3d_p3_clip.txt
  |_16point_mean_limb_scaled_max_min.csv #16 points normalize by (x-min) / (max-min)
  ```

  After set up Human3.6m dataset following its illustration and download above training/testing list. You should update “root_folder” paths in **CAFFE_ROOT/examples/.../*.prototxt** for images and annotation director. 

+ MPII

  We crop and square single person from  all images and update 2d annotation in train_h36m.txt (resort points according to order of Human3.6m points).

  ```
  mkdir data/MPII
  cd data/MPII
  wget -v https://drive.google.com/open?id=16gQJvf4wHLEconStLOh5Y7EzcnBUhoM-
  tar -xzvf MPII_square.tar.gz
  rm -f MPII_square.tar.gz
  ```

  

### Training

#### Offline Phase

Our model consists of two cascade modules, so training phase can be divided into the following setps:

```
cd CAFFE_ROOT
```

1. Pre-train the *2D pose sub-network* with MPII. You can follow [CPM](https://arxiv.org/abs/1602.00134) or [Hourglass](https://arxiv.org/abs/1603.06937) or other 2D pose estimation method. We provide pretrained [CPM-caffemodel](https://drive.google.com/open?id=1fUfC7NWFbWOPmDPFBGu3iPFK8nO6PFIM). Please put it into *CAFFE_ROOT/models/*.

2. Train *2D-to-3D pose transformer module* with Human3.6M. And we fix the parameters of the *2D pose sub-network*. The corresponding prototxt file is in *examples/2D_to_3D/bilstm.prototxt*. 

   ```
   sh examples/2D_to_3D/train.sh
   ```

3. To train *3D-to-2D* pose projector module, we fix above module weights. And we need in the wild 2D Pose dataset to help training (we choose MPII).

   ```sh
   sh examples/3D_to_2D/train.sh
   ```

4. Fine-tune the whole model jointly. We provide [trained model and coarse prediction of Protocol I and Protocol III](https://drive.google.com/open?id=1DS50Na6fbTaG-mHVzFbINx9AE5Wgpqa6).

   ```sh
   sh examples/finetune_whole/train.sh
   ```

5. Model extension: Add rand mask to relieve model bias. We provide corresponding model files in *examples/mask3d*.

   ```sh
   sh examples/mask3d/train.sh
   ```



### Model Inference

3D-to-2D project module are initialized from the well trained model, and they will be updated by minimizing the difference between the predicted 2D pose and projected 2D pose.

+ Inference with [provided models](https://drive.google.com/open?id=1dMuPuD_JdHuMIMapwE2DwgJ2IGK04xhQ)

  ```shell
  # Step1: Download the trained model
  cd PROJECT_ROOT
  mkdir models
  cd models
  wget -v https://drive.google.com/open?id=1dMuPuD_JdHuMIMapwE2DwgJ2IGK04xhQ
  unzip model_extension_mask3d.zip
  rm -r model_extension_mask3d.zip
  cd ../
  
  # Step2: save coarse 3D prediction
  cd test
  # change 'data_root' in test_human16.sh 
  # change 'root_folder' in template_16_merge.prototxt
  # test_human16.sh [$1 deploy.prototxt] [$2 trained model] [$3 save dir] [$4 batchsize]
  sh test_human16.sh . ../models/model_extension_mask3d/mask3d_iter_400000.caffemodel mask3d 5
  
  # Step3: online refine 3D pose prediction
  # protocal: 1/3 , default is 1
  # pose2d: ours/hourglass/gt, default is ours
  # coarse_3d: saved results in Sept2
  python pred_v2.py --trained_model ../models/model_extension_mask3d/mask3d-400000.pkl --protocol 1 --data_dir /data/h36m/ --coarse_3d ../test/mask3d --save srr_results --pose2d hourglass
  ```

  

+ Inference with [Our2d-model](https://drive.google.com/open?id=19kTyttzUnm_1_7HEwoNKCXPP2QVo_zcK)

  ```shell
  # Maybe you want to predict 2d.
  # The model we use to predict 2d pose is similar with our 3dpredict model without ssl module.
  # Or you can use Hourglass(https://github.com/princeton-vl/pose-hg-demo) to predict 2d pose
  
  # Step1.1: Download the trained merge model
  cd PROJECT_ROOT
  mkdir models && cd models
  wget -v https://drive.google.com/open?id=19kTyttzUnm_1_7HEwoNKCXPP2QVo_zcK
  unzip our2d.zip
  rm -r our2d.zip
  # move 2d prototxt to PROJECT_ROOT/test/
  mv our2d/2d ../test/
  cd ../
  
  # Step1.2: save 2D prediction
  cd test
  # change 'data_root' in test_human16.sh 
  # change 'root_folder' in 2d/template_16_merge.prototxt
  # test_human16.sh [$1 deploy.prototxt] [$2 trained model] [$3 save dir] [$4 batchsize]
  sh test_human16.sh 2d/ ../models/our2d/2d_iter_800000.caffemodel our2d 5
  # replace predict 2d pose in data dir or change data_dir in tensorflow/pred_v2.py
  mv our2d /data/h36m/ours_2d/bilstm2d-p1-800000
  
  
  # Step2 is same with above
  
  
  # Step3: online refine 3D pose prediction
  # protocal: 1/3 , default is 1
  # pose2d: ours/hourglass/gt, default is ours
  # coarse_3d: saved results in Sept2
  python pred_v2.py --trained_model ../models/model_extension_mask3d/mask3d-400000.pkl --protocol 1 --data_dir /data/h36m/ --coarse_3d ../test/mask3d --save srr_results --pose2d ours
  ```

  

+ Inference with yourself

  Only difference is that you should transfer caffemodel of 3D-to-2D project module to pkl file. We provide *gen_refinepkl.py* in tools/.

  ```sh
  # Follow above Step1~2 to produce coarse 3d prediction and 2d pose.
  # transfer caffemodel of SRR module to python .pkl file
  python tools/gen_refinepkl.py CAFFE_ROOT CAFFEMODEL_DIR --pkl_dir model.pkl
  
  # online refine 3D pose prediction
  python pred_v2.py --trained_model model.pkl
  ```

  

+ Evaluation

  ```shell
  # Print MPJP 
  run tools/eval_h36m.m
  
  # Visualization of 2dpose/ 3d gt pose/ 3d coarse pose/ 3d refine pose
  # Please change data_root in visualization.m before running
  run visualization.m
  ```



### Citation

```
@article{wang20193d,
  title={3D Human Pose Machines with Self-supervised Learning},
  author={Wang, Keze and Lin, Liang and Jiang, Chenhan and Qian, Chen and Wei, Pengxu},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
  publisher={IEEE}
}
```
