#!/usr/bin/env bash

data_root="/data/h36m/gt/test/"
phase=test
num_action=15
prototxt_fn=$1/template_16_merge.prototxt

if [ ! -f $prototxt ]; 
then
	echo "Error, prototxt not found"
  echo $prototxt_fn
  exit
fi

model_fn=$2
tmp_proto_fn="./${phase}_tmp.prototxt"
tmp_proto_fn_2="./${phase}_tmp_2.prototxt"
result=$3/result
batch=$4

aid=1
#for ((aid = 1; aid <= $num_action; ++aid))
while [ $aid -le 15 ]
do
  result_file=$result$aid
  echo $result_file

  data_folder=$data_root$phase$aid
  data_folder=${data_folder}".txt"
  SAMPLE_NUM=`cat ${data_folder} | wc -l`

  echo sed "s#DATA_FOLDER#${data_folder}#g"  $prototxt_fn  \> $tmp_proto_fn
  sed "s#DATA_FOLDER#${data_folder}#g"  $prototxt_fn  > $tmp_proto_fn
  sed "s#RESULT#${result_file}#g"  $tmp_proto_fn  > $tmp_proto_fn_2
  sed "s#SAMPLE_NUM#${SAMPLE_NUM}#g"  $tmp_proto_fn_2  > $tmp_proto_fn
  sed "s#BATCH#${batch}#g"  $tmp_proto_fn  > $tmp_proto_fn_2

  
  ../../../build/tools/caffe test -gpu=0 -model=$tmp_proto_fn_2 \
  -weights=$model_fn \
  -iterations=`expr $SAMPLE_NUM / $batch` 
  #2>&1 | tee $1/log/${phase}_${aid}.log
  aid=$(($aid + 1))
done
