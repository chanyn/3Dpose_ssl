#!/bin/bash

TOOLS=../../build/tools
WEIGHTS=../2D_to_3D/params/
MEDOL=.

$TOOLS/caffe train -gpu=0 -solver=$MODEL/solver.prototxt -weights=$WEIGHTS/bilstm_iter_300000.caffemodel 2>&1 | tee -a train.log
