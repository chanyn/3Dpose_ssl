// Copyright DDK 2015 

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>

#include "caffe/my_layers/pose_heat_map_loss_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/my_layers/global_variables.hpp"
// #include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK(this->layer_param_.has_pose_heat_map_loss_param());
  const PoseHeatMapLossParameter pose_heat_map_loss_param = 
      this->layer_param_.pose_heat_map_loss_param();
  CHECK(pose_heat_map_loss_param.has_fg_eof());
  CHECK(pose_heat_map_loss_param.has_bg_eof());
  CHECK(pose_heat_map_loss_param.has_ratio());
  CHECK(pose_heat_map_loss_param.has_loss_emphase_type());
  // CHECK(pose_heat_map_loss_param.has_key_point_num());

  this->fg_eof_ = pose_heat_map_loss_param.fg_eof();
  this->bg_eof_ = pose_heat_map_loss_param.bg_eof();
  this->ratio_  = pose_heat_map_loss_param.ratio();
  this->loss_emphase_type_ = pose_heat_map_loss_param.loss_emphase_type();
  // this->key_point_num_ = pose_heat_map_loss_param.key_point_num();

  CHECK_GT(this->ratio_, 0.);
  CHECK_LT(this->ratio_, 1.);

  LOG(INFO) << ", loss_emphase_type: " << this->loss_emphase_type_
      << ", fg_eof: " << this->fg_eof_ 
      << ", bg_eof: " << this->bg_eof_
      << ", ratio: " << this->ratio_;
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // call the parent-class function
  LossLayer<Dtype>::Reshape(bottom, top);
  // heat maps
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  // heat map mask: indicate which heat map is valid or needs bp
  if(bottom.size() == 3) {
    CHECK_EQ(bottom[0]->num(), bottom[2]->num())
        << "The data and label should have the same number.";
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels())
        << "Inputs must have the same dimension.";  
    CHECK_EQ(bottom[2]->channels(), bottom[2]->count() / bottom[2]->num());
  }

  this->key_point_num_ = bottom[0]->channels();
  this->heat_num_ = bottom[0]->width() * bottom[0]->height();
  CHECK_GT(this->key_point_num_, 0);

  // Reference to `src/caffe/layers/dropout_layer.cpp` or `src/caffe/neuron_layers.hpp`
  rand_vec_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  // UINT_MAX(unsigned int): 4294967295, uint_thres_: 42949672
  this->uint_thres_ = static_cast<unsigned int>(UINT_MAX * this->ratio_);

  // 
  this->diff_.ReshapeLike(*bottom[0]);
}

/**
 * @brief bottom[0] is predicted blob, bottom[1] is ground truth blob
 * But sometimes, bottom[1] may be another predicted blob (as in siamese network)
 * So, ...
*/
template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
 /*__asm__("int $3");
  
  for(int i =0; i < bottom[0]->num();i++){
    for( int j=0;j < bottom[0]->channels();j++){
      for(int k =0; k<bottom[0]->height(); k++){
        for(int l =0; l< bottom[0]->width();l++){
          if(bottom[1]->data_at(i,j,k,l) != 0.0){
            __asm__("int $3");
            LOG(INFO)<<bottom[1]->data_at(i,j,k,l);
            LOG(INFO)<<bottom[0]->data_at(i,j,k,l);
          }
        }
      }
    }
  }*/
    const int count = bottom[0]->count();
  // diff
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      this->diff_.mutable_cpu_data()
  );
  // loss
  Dtype dot = caffe_cpu_dot(
      count, 
      this->diff_.cpu_data(), 
      this->diff_.cpu_data()
  );
  Dtype loss = dot / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::PrintLoss() 
{
  Dtype batch_loss = caffe_cpu_dot(
      this->diff_.count(), 
      this->diff_.cpu_data(), 
      this->diff_.cpu_data()
  );
  const Dtype per_frame_loss = batch_loss / Dtype(this->diff_.num() + 0.);
  const Dtype per_heat_map_loss = per_frame_loss / Dtype(this->key_point_num_ + 0.);

  LOG(INFO) << "iter: " << GlobalVars::caffe_iter() << ", learn_lr: " << GlobalVars::learn_lr() << "euclidean loss (batch): " << batch_loss <<"euclidean loss (frame): " << per_frame_loss;
  LOG(INFO) << "euclidean loss (joint): " << per_heat_map_loss<<"loss_emphase_type: " << loss_emphase_type_;
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::ComputesHeatMapLoss(
    const vector<Blob<Dtype>*>& bottom) 
{
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int heat_map_channels = bottom[0]->channels();
  CHECK_LE(this->key_point_num_, heat_map_channels);

  /* Get data&labels pointers */
  const Dtype* bottom_ground_truths_ptr = bottom[1]->cpu_data();
  Dtype* diff_ptr = this->diff_.mutable_cpu_data();

  /* 0: default, means do nothing */
  /* Only consider ground truths */
  if(this->loss_emphase_type_ == 1) {
    for(int idx = 0; idx < count; idx++) {
      if(bottom_ground_truths_ptr[idx] == 0) {
        diff_ptr[idx] = Dtype(0);
      }
    }

  /* Consider backgrounds and ground truths(but scale them) */
  } else if(this->loss_emphase_type_ == 2) {
    for(int idx = 0; idx < count; idx++) {
      if(bottom_ground_truths_ptr[idx] != 0) {
        diff_ptr[idx] *= this->fg_eof_;
      }
    }

  /* Consider a littel bit of  backgrounds and ground truths(but scale them) */
  } else if(this->loss_emphase_type_ == 3 ||
      this->loss_emphase_type_ == 4) 
  {
    // get foreground eof 
    Dtype fg_eof2 = this->loss_emphase_type_ == 4 
        ? this->fg_eof_ : Dtype(1);
    // produce the random number
    unsigned int* rand_mask = rand_vec_.mutable_cpu_data();

    // get offset
    int hm_offset = 0;
    for(int np = 0; np < num; np++) {
      for(int kpn = 0; kpn < heat_map_channels; kpn++) {
        /* Create random numbers */
        caffe_rng_bernoulli(this->heat_num_, this->ratio_, rand_mask);

        for(int hn = 0; hn < this->heat_num_; hn++) {
          // Remove most of backgrounds because of ratio_
          if(rand_mask[hn] != 0
              || bottom_ground_truths_ptr[hm_offset] != 0) 
          {
            /* Only scale the ground truths */
            if(bottom_ground_truths_ptr[hm_offset] != 0) {
              diff_ptr[hm_offset] *= fg_eof2;  
            } else {
              // the backgrounds either keep the same 
              // (since bg_eof_ is always 1), or reset to be zero
              diff_ptr[hm_offset] *= this->bg_eof_;
            }
          } else {
            diff_ptr[hm_offset] = Dtype(0);
          }

          // increase the offset by one
          hm_offset++;
        }
      }
    }
    CHECK_EQ(hm_offset, count) << "does not match the size of each blob in bottom";
  } else {
    if(this->loss_emphase_type_) {
      LOG(INFO) << "loss_emphase_type_: " << this->loss_emphase_type_;
      NOT_IMPLEMENTED;
    }
  }

}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::CopyDiff(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // mask
  if(bottom.size() == 3) {
    const Dtype* mask = bottom[2]->cpu_data();
    const int batch_num = bottom[2]->num();
    const int sub_count = bottom[2]->count() / batch_num;
    CHECK_EQ(sub_count, this->key_point_num_);
    
    // Use mask to filter out some outliers
    for(int n = 0; n < batch_num; n++) {
      for(int sc = 0; sc < sub_count; sc++) {
        const int mask_offset = bottom[2]->offset(n, sc);
        const int diff_offset = this->diff_.offset(n, sc);
        caffe_scal(
            this->heat_num_, 
            mask[mask_offset],
            this->diff_.mutable_cpu_data() + diff_offset
        );
      }
    }
  }

  // Copy
  for (int idx = 0; idx < 2; ++idx) {
    if (propagate_down[idx]) {
      const Dtype sign = (idx == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[idx]->num();
  
      caffe_cpu_axpby(
          bottom[idx]->count(),                 // count
          alpha,                                // alpha
          this->diff_.cpu_data(),               // a
          Dtype(0),                             // beta
          bottom[idx]->mutable_cpu_diff()       // b
      );  
    }
  }
}

/**
 * @brief bottom[0] is predicted blob, bottom[1] is ground truth blob
 * But sometimes, bottom[1] may be another predicted blob (as in siamese network)
*/
template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // Compute correspoding with loss_emphase_type
  this->ComputesHeatMapLoss(bottom);
  // Copy
  this->CopyDiff(top, propagate_down, bottom);

  // Print loss
  this->PrintLoss();
}

#ifdef CPU_ONLY
STUB_GPU(PoseHeatMapLossLayer);
#endif

INSTANTIATE_CLASS(PoseHeatMapLossLayer);
REGISTER_LAYER_CLASS(PoseHeatMapLoss);

}  // namespace caffe