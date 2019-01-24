// Copyright 2015 

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <functional>
#include <utility>
#include "boost/algorithm/string.hpp"

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
// #include "caffe/vision_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/my_layers/global_variables.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  CHECK(this->layer_param_.has_pose_pdj_accuracy_param());
  // __asm__("int $3");
  const PosePDJAccuracyParameter pose_pdj_accuracy_param = 
      this->layer_param_.pose_pdj_accuracy_param();  
  CHECK(pose_pdj_accuracy_param.has_acc_factor());
  CHECK(pose_pdj_accuracy_param.has_acc_factor_num());
  CHECK(pose_pdj_accuracy_param.has_images_num());
  CHECK(pose_pdj_accuracy_param.has_acc_path());
  CHECK(pose_pdj_accuracy_param.has_acc_name());
  CHECK(pose_pdj_accuracy_param.has_log_name());
  CHECK(pose_pdj_accuracy_param.has_shoulder_id());
  CHECK(pose_pdj_accuracy_param.has_hip_id());
  // CHECK(pose_pdj_accuracy_param.has_label_num());
  // get config variables
  this->images_itemid_ = 0;

  // this->label_num_ = pose_pdj_accuracy_param.label_num();
  this->images_num_ = pose_pdj_accuracy_param.images_num();
  const float acc_factor = pose_pdj_accuracy_param.acc_factor();
  this->acc_factor_ = acc_factor;
  this->acc_factor_num_ = pose_pdj_accuracy_param.acc_factor_num();
  this->acc_path_ = pose_pdj_accuracy_param.acc_path();
  this->acc_name_ = pose_pdj_accuracy_param.acc_name();
  this->log_name_ = pose_pdj_accuracy_param.log_name();
  this->shoulder_id_ = pose_pdj_accuracy_param.shoulder_id();
  this->hip_id_ = pose_pdj_accuracy_param.hip_id();
  // accuracy log file
  // if(IsDiretory(this->acc_path_) {}
  // CreateDir(this->acc_path_);
  this->acc_file_ = this->acc_path_ + this->acc_name_;
  this->log_file_ = this->acc_path_ + this->log_name_;
 
  LOG(INFO) << "acc_path: " << this->acc_path_;
  LOG(INFO) << "acc_name: " << this->acc_name_;
  LOG(INFO) << "acc_file: " << this->acc_file_;
  LOG(INFO) << "log_file: " << this->log_file_;
  LOG(INFO) << "acc_factor: " << this->acc_factor_;

  CHECK_GT(acc_factor, 0.);
  CHECK_LE(acc_factor, 1.);
  CHECK_GT(this->acc_factor_num_, 0);
  CHECK_LT(this->acc_factor_num_, int(1.0 / this->acc_factor_));

  // get threshold for PDJ
  LOG(INFO) << "accuracy factors below: ";
  for (int idx = 1; idx <= this->acc_factor_num_; idx++) {
    this->acc_factors_.push_back(idx * acc_factor);
    LOG(INFO) << idx << ": accuracy factor: " << acc_factor
        << ", accuracy threshold: " << idx * acc_factor;
  }
  // initialize
  this->label_num_ = bottom[0]->channels();
  this->key_point_num_ = this->label_num_ / 2;
  this->initAccFactors();

  LOG(INFO) << "acc_factor: " << this->acc_factor_;
  LOG(INFO) << "acc_factors_num: " << this->acc_factor_num_;
  LOG(INFO) << "images_num: " << this->images_num_;
  LOG(INFO) << "images_itemid: " << this->images_itemid_;
  LOG(INFO) << "label_num: " << this->label_num_;
  LOG(INFO) << "key_point_num: " << this->key_point_num_;
}

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const std::string err_str = "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << err_str;
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << err_str;
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << err_str;
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << err_str;
  CHECK_EQ(bottom[0]->channels(), bottom[0]->count() / bottom[0]->num()) << err_str;

  // check labels' number
  this->label_num_ = bottom[0]->channels();
  this->key_point_num_ = this->label_num_ / 2;
  // check
  CHECK_GT(this->label_num_, 0);
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2) << err_str;
  CHECK_GE(this->shoulder_id_, 0);
  CHECK_LE(this->shoulder_id_, this->key_point_num_ - 1);
  CHECK_GE(this->hip_id_, 0);
  CHECK_LE(this->hip_id_, this->key_point_num_ - 1);

  // diff_
  this->diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  // top blob: record the accuracy ?
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::initAccFactors() {
	// clear
  if(!this->accuracies_.empty()) {
    for (int i = 0; i < this->acc_factor_num_; i++) {
      this->accuracies_[i].clear();
    }
    this->accuracies_.clear();
  }
	CHECK_EQ(this->accuracies_.size(), 0) << "invalid accuracies variable, when reset them";

	for (int idx = 0; idx < this->acc_factor_num_; idx++) {
	  std::vector<float> acc_temp;
	  for(int sln = 0; sln < this->key_point_num_; sln++) {
	    acc_temp.push_back(0.);
	  }
	  this->accuracies_.push_back(acc_temp);
	}
}

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::InitQuantization() {
  // check
  CHECK_GE(this->images_itemid_, this->images_num_);
	// re-init
  this->initAccFactors();
  // reset images_itemid_ to be zero
  this->images_itemid_ = 0;
}

// shoulder_id & hip_id start from zero 
template<typename Dtype>
void PosePDJAccuracyLayer<Dtype>::CalAccPerImage(
    const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr) 
{
  const Dtype Zero = Dtype(0);
  const int hip_index = this->hip_id_ * 2;
  const int shoulder_index = this->shoulder_id_ * 2;

  // shoulder and hip indices&coordinates
  Dtype shoulder_x = gt_coords_ptr[shoulder_index];
  Dtype shoulder_y = gt_coords_ptr[shoulder_index + 1];
  Dtype hip_x = gt_coords_ptr[hip_index];
  Dtype hip_y = gt_coords_ptr[hip_index + 1];
  // if the ground truth is not valid, then ignore it
  if(shoulder_x < Zero || shoulder_y < Zero || hip_x < Zero || hip_y < Zero) return;

  const Dtype diff_x = shoulder_x - hip_x;
  const Dtype diff_y = shoulder_y - hip_y;
  const Dtype gt_dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);

  int index = 0;
  //Dtype pred_dist[this->key_point_num_];
  Dtype *pred_dist = new Dtype[this->key_point_num_];

  for (int idx = 0; idx < this->label_num_; idx += 2) {
    // ground true x and y
    const Dtype gt_x = gt_coords_ptr[idx];
    const Dtype gt_y = gt_coords_ptr[idx + 1];
    if(gt_x < Zero || gt_y < Zero) {
      pred_dist[index++] = gt_dist + 1;
      continue;
    }
    // predicted x and y
    Dtype pred_x = pred_coords_ptr[idx];
    Dtype pred_y = pred_coords_ptr[idx + 1];
    
    // distance between prediction and ground truth
    const Dtype diff_x2 = pred_x - gt_x;
    const Dtype diff_y2 = pred_y - gt_y;
    pred_dist[index++] = std::sqrt(diff_x2 * diff_x2 + diff_y2 * diff_y2);
  }

  // Count the correct ones, if the distance is less than 
  // some fraction of the distance between hip and shoulder.
  for (int afn = 0; afn < this->acc_factor_num_; afn++) {
    const Dtype threshold = gt_dist * this->acc_factors_[afn];
    for (int idx = 0; idx < this->key_point_num_; idx++) {
      if (pred_dist[idx] <= threshold) {
        this->accuracies_[afn][idx]++;
      }
    }
  }
  delete[] pred_dist;
}

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::WriteResults(const float total_accuracies[]) {
  std::ofstream acc_fhd;
  acc_fhd.open(this->acc_file_.c_str(), ios::out | ios::app);
  CHECK(acc_fhd);

  acc_fhd << GlobalVars::SpiltCodeBoundWithStellate();
  acc_fhd << GlobalVars::SpiltCodeBoundWithStellate() << std::endl;
  acc_fhd << "iter: " << GlobalVars::caffe_iter() << std::endl;
  acc_fhd << "learn_lr: " << GlobalVars::learn_lr() << std::endl;
  
  for(int afn = 0; afn < this->acc_factor_num_; afn++) {
    acc_fhd << "acc_factor: " << acc_factors_[afn] << " " << std::endl;
    acc_fhd << "accuracy: " << total_accuracies[afn] << std::endl;
    for (int lnh = 0; lnh < this->key_point_num_ - 1; lnh++) {
      acc_fhd << lnh + 1 << "th: " << accuracies_[afn][lnh] << ", ";
      if((lnh + 1) % 10 == 0) {
        acc_fhd << std::endl;
      }
    }
    acc_fhd << this->key_point_num_ << "th: " 
        << this->accuracies_[afn][this->key_point_num_ - 1];
    acc_fhd << std::endl;
  }

  acc_fhd << std::endl << std::endl;
  acc_fhd.close();
}

template<typename Dtype>
void PosePDJAccuracyLayer<Dtype>::QuanFinalResults() {
  // statistically 
  //float total_accuracies[this->acc_factor_num_];
  float *total_accuracies = new float[this->acc_factor_num_];
  for(int afn = 0; afn < this->acc_factor_num_; afn++) {
    total_accuracies[afn] = 0.;

    for(int lnh = 0; lnh < this->key_point_num_; lnh++) {  
      this->accuracies_[afn][lnh] /= (this->images_num_ + 0.);
      total_accuracies[afn] += accuracies_[afn][lnh];
    }

    total_accuracies[afn] /= (this->key_point_num_ + 0.);
  }	

  // save
  this->WriteResults(total_accuracies);
  delete []total_accuracies;
}

template<typename Dtype>
void PosePDJAccuracyLayer<Dtype>::Quantization(
    const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr, const int num) 
{
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2) 
      << "invalid label_num: " << this->label_num_ << ", and s_label_num: " << this->key_point_num_;

  for(int idx = 0; idx < num; idx++) {
    const Dtype* gt_coords_ptr2   = gt_coords_ptr + this->label_num_ * idx;
    const Dtype* pred_coords_ptr2 = pred_coords_ptr + this->label_num_ * idx;

    if(this->images_itemid_ < this->images_num_) {      
      this->CalAccPerImage(pred_coords_ptr2, gt_coords_ptr2);
      this->images_itemid_++;
      LOG(INFO) << "images_itemid: " << this->images_itemid_ << " (" << this->images_num_ << ")";
    } else {
      LOG(INFO) << "ready for writing the accuracy results...";
      break;
    }
  }

  if(this->images_itemid_ >= this->images_num_) {
    // Record final results
    this->QuanFinalResults(); 
    // Initialize
    this->InitQuantization();
  }
}

template <typename Dtype>
void PosePDJAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int count = bottom[0]->count();
	const int num = bottom[0]->num(); 
  CHECK_EQ(count, num * this->label_num_);
  
	caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), this->diff_.mutable_cpu_data());
	Dtype loss = caffe_cpu_dot(count, this->diff_.cpu_data(), this->diff_.cpu_data());
  loss /= Dtype(num + 0.);
  top[0]->mutable_cpu_data()[0] = loss;
  LOG(INFO) << "iter: " << GlobalVars::caffe_iter() << ", loss: " <<  loss;

  std::ofstream log_fhd;
  log_fhd.open(this->log_file_.c_str(), ios::out | ios::app);
  log_fhd << "iter: " << GlobalVars::caffe_iter() << ", loss: " <<  loss << std::endl;
  log_fhd.close();

  const Dtype* gt_coords_ptr = bottom[1]->cpu_data();
  const Dtype* pred_coords_ptr = bottom[0]->cpu_data();
  this->Quantization(pred_coords_ptr, gt_coords_ptr, num);
}

INSTANTIATE_CLASS(PosePDJAccuracyLayer);
REGISTER_LAYER_CLASS(PosePDJAccuracy);

}  // namespace caffe