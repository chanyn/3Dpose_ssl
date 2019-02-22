#!/bin/bash

TOOLS=../../build/tools
WEIGHTS=../finetune_whole/params/
MEDOL=.

$TOOLS/caffe train -gpu=0 -solver=$MODEL/solver.prototxt -weights=$WEIGHTS/finetune_whole.caffemodel 2>&1 | tee -a train.log
