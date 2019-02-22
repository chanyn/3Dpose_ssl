#!/bin/bash

TOOLS=../../build/tools
WEIGHTS=../3D_to_2D/params/
MEDOL=.

$TOOLS/caffe train -gpu=0 -solver=$MODEL/solver.prototxt -weights=$WEIGHTS/addsrr_iter_300000.caffemodel 2>&1 | tee -a train.log
