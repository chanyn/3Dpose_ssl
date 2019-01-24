// Copyright 2015 DDK (dongdk.sysu@foxmail.com)

#include <algorithm>
#include <cfloat>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;
struct bodyparttemplate
{
  int heatmapindice;
  string parent;
  int xmove;
  int ymove;
  int width;
  int height;
};
template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_pose_coords_from_heat_maps_param());
  const CoordsFromHeatMapsParameter pose_coords_from_heat_maps_param = 
      this->layer_param_.pose_coords_from_heat_maps_param();

  CHECK(pose_coords_from_heat_maps_param.has_heat_map_a());
  CHECK(pose_coords_from_heat_maps_param.has_heat_map_b());

  this->heat_map_a_  = pose_coords_from_heat_maps_param.heat_map_a();
  this->heat_map_b_  = pose_coords_from_heat_maps_param.heat_map_b();

  this->heat_channels_ = bottom[0]->channels();
  this->heat_width_ = bottom[0]->width();
  this->heat_height_ = bottom[0]->height();
  this->batch_num_ = bottom[0]->num();
  this->heat_count_ = bottom[0]->count();
  this->heat_num_ = this->heat_width_ * this->heat_height_;

  this->per_heatmap_ch_ = pose_coords_from_heat_maps_param.per_heatmap_ch();
  this->key_point_num_ = this->heat_channels_ / this->per_heatmap_ch_;
  this->label_num_ = (this->per_heatmap_ch_ == 1) ? this->key_point_num_ * 2 : this->key_point_num_ * 3;
  LOG(INFO) << "*************************************************";
  LOG(INFO) << "key_point_num: " << this->key_point_num_;
  LOG(INFO) << "label_num: " << this->label_num_;
  LOG(INFO) << "heat_width: " << this->heat_width_;
  LOG(INFO) << "heat_height: " << this->heat_height_;
  LOG(INFO) << "batch_num: " << this->batch_num_;
  // LOG(INFO) << "heat_map_a: " << this->heat_map_a_;
  // LOG(INFO) << "heat_map_b: " << this->heat_map_b_;
  LOG(INFO) << "**************************************************";
}


template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  this->heat_channels_ = bottom[0]->channels();
  this->heat_width_ = bottom[0]->width();
  this->heat_height_ = bottom[0]->height();
  this->batch_num_ = bottom[0]->num();
  this->heat_count_ = bottom[0]->count();
  this->heat_num_ = this->heat_width_ * this->heat_height_;
  this->per_heatmap_ch_ = this->per_heatmap_ch_;
  this->key_point_num_ = this->heat_channels_ / this->per_heatmap_ch_;
  this->label_num_ = (this->per_heatmap_ch_ == 1) ? this->key_point_num_ * 2 : this->key_point_num_ * 3;

  // top blob
  top[0]->Reshape(this->batch_num_, this->label_num_, 1, 1);
  // prefetch_coordinates_labels
  this->prefetch_coordinates_labels_.Reshape(
      this->batch_num_, this->label_num_, 1, 1);

  if(top.size() > 1) {
    // max score of heat map for each part/joint
    CHECK_EQ(top.size(), 2);
    top[1]->Reshape(
      this->batch_num_, this->key_point_num_, 1, 1);
    // corresponding scores
    this->prefetch_coordinates_scores_.Reshape(
        this->batch_num_, this->key_point_num_, 1, 1);
  }
}

template <typename Dtype> 
void CoordsFromHeatMapsLayer<Dtype>::CreateCoordsFromHeatMap(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const Dtype* heat_map_ptr = bottom[0]->cpu_data();

  // coordinates ptr
  // Need to be reset or not ?
  Dtype* coordinates_ptr = 
      this->prefetch_coordinates_labels_.mutable_cpu_data();
  // score ptr
  Dtype* scores_ptr = NULL;
  bool has_top_score = false;
  if(top.size() == 2) {
    has_top_score = true;
    scores_ptr = this->prefetch_coordinates_scores_.mutable_cpu_data();
  }

  int max_val_idx, max_kpn_idx, last_channel_max;
  int pred_x_idx, pred_y_idx, pred_z_idx;
  int scores_offset = 0;
  int heat_map_offset = 0;
  int coordinates_offset = 0;
  Dtype part_max_val;

  this->heat_channels_ = bottom[0]->channels();
  std::cout << "channels" << this->heat_channels_<<std::endl;
  std::cout << "key_point_num_" << this->key_point_num_<<std::endl;
  CHECK_LE(this->key_point_num_, this->heat_channels_);
  
  if(this->per_heatmap_ch_ == 1) {
    for(int item_id = 0; item_id < this->batch_num_; item_id++) {
      for(int kpn = 0; kpn < this->heat_channels_; kpn++) {
        // Initliaze
        part_max_val = Dtype(-FLT_MAX);
        max_val_idx = 0;

        for(int hn = 0; hn < this->heat_num_; hn++) {
          // Find max value and its corresponding index
          if(part_max_val < heat_map_ptr[heat_map_offset]) {
            part_max_val = heat_map_ptr[heat_map_offset];
            max_val_idx = hn;
          }

          // Increase the step/heat_map_offset
          heat_map_offset++;
        }
        // coordinate from heat map
        pred_x_idx = max_val_idx % this->heat_width_;
        pred_y_idx = max_val_idx / this->heat_width_;
        // coordinate from image
        pred_x_idx = pred_x_idx / this->heat_width_;
        pred_y_idx = pred_y_idx / this->heat_height_;
        // pred_x_idx = pred_x_idx * this->heat_map_a_ + this->heat_map_b_;
        // pred_y_idx = pred_y_idx * this->heat_map_a_ + this->heat_map_b_;

        // Set index value (this is the initial predicted coordinates)
        coordinates_ptr[coordinates_offset++] = pred_x_idx;
        coordinates_ptr[coordinates_offset++] = pred_y_idx;
        if(has_top_score) {
          // Record the corresponding sroce
          scores_ptr[scores_offset++] = part_max_val;
        }
      }
    }
  }
  else {
    for(int item_id = 0; item_id < this->batch_num_; item_id++) {
      for(int kpn = 0; kpn < this->key_point_num_; kpn++ ) {
        // Initliaze
        part_max_val = Dtype(-FLT_MAX);
        max_val_idx = 0;
        max_kpn_idx = 0;
        last_channel_max = 0;
        for(int pn = 0; pn < this->per_heatmap_ch_; pn++) {
          for(int hn = 0; hn < this->heat_num_; hn++) {
            // Find max value and its corresponding index
            if(part_max_val < heat_map_ptr[heat_map_offset]) {
              part_max_val = heat_map_ptr[heat_map_offset];
              max_val_idx = hn;
            }
           // Increase the step/heat_map_offset
            heat_map_offset++;
            if( heat_map_ptr[pn * this->heat_num_+ hn] > last_channel_max) {
              last_channel_max = heat_map_ptr[pn * this->heat_num_+ hn];
              max_kpn_idx = pn;
            }
          }
        }
        // coordinate from heat map
        LOG(INFO)<<"max_val_idx:"<<max_val_idx<<" max_kpn_idx:"<<max_kpn_idx;
        pred_x_idx = max_val_idx % this->heat_width_;
        pred_y_idx = max_val_idx / this->heat_width_;
        // coordinate from image
        pred_x_idx = pred_x_idx / this->heat_width_;
        pred_y_idx = pred_y_idx / this->heat_height_;
        pred_z_idx = max_kpn_idx / this->per_heatmap_ch_;

         // Set index value (this is the initial predicted coordinates)
        coordinates_ptr[coordinates_offset++] = pred_x_idx;
        coordinates_ptr[coordinates_offset++] = pred_y_idx;
        coordinates_ptr[coordinates_offset++] = pred_z_idx;
        if(has_top_score) {
          // Record the corresponding sroce
          scores_ptr[scores_offset++] = part_max_val;
        }
      }
    }
  }

  CHECK_EQ(heat_map_offset, bottom[0]->count());
  CHECK_EQ(coordinates_offset, this->prefetch_coordinates_labels_.count());
  if(has_top_score) {
    CHECK_EQ(scores_offset, this->prefetch_coordinates_scores_.count());
  }
}

template <typename Dtype> 
void CoordsFromHeatMapsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Get coordinates
  CreateCoordsFromHeatMap(bottom, top);
  // Copy the preliminary&predicted coordinates labels
  caffe_copy(
      top[0]->count(), 
      this->prefetch_coordinates_labels_.cpu_data(), 
      top[0]->mutable_cpu_data()
  );

  if(top.size() == 2) {
    // Copy the corresponding maximized scores or response values
    caffe_copy(
        top[1]->count(), 
        this->prefetch_coordinates_scores_.cpu_data(), 
        top[1]->mutable_cpu_data()
    );
  }
}


template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CoordsFromHeatMapsLayer);
#endif

INSTANTIATE_CLASS(CoordsFromHeatMapsLayer);
REGISTER_LAYER_CLASS(CoordsFromHeatMaps);

}  // namespace caffe