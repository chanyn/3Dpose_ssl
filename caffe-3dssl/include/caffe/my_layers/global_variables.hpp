#ifndef CAFFE_UTIL_GLOBALVARS_HPP_
#define CAFFE_UTIL_GLOBALVARS_HPP_

#include <boost/shared_ptr.hpp>

#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <cmath>
#include <math.h>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
namespace caffe {

// A singleton class to hold common global variables
// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
class GlobalVars {
 public:
  ~GlobalVars();
  inline static GlobalVars& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new GlobalVars());
    }
    return *singleton_;
  }

  // ####################################################################

  // phase
  inline static void set_phase(Phase phase) { Get().phase_ = phase; }
  // TRAIN or TEST
  inline static Phase phase() { return Get().phase_; }

  // caffe_iter
  inline static void set_caffe_iter(const int caffe_iter) { 
    Get().caffe_iter_ = caffe_iter; 
  }  
  inline static int caffe_iter() { return Get().caffe_iter_; }

  // learn_lr
  inline static void set_learn_lr(const float learn_lr) { 
    Get().learn_lr_ = learn_lr; 
    // LOG(INFO) << "learn_lr: " << Get().learn_lr_;
    // exit(1);
  }  
  inline static float learn_lr() { return Get().learn_lr_; }

  // objidxs
  inline static void set_objidxs(const std::vector<std::string>& objidxs) {
    if(!Get().objidxs_.empty()) {
      Get().objidxs_.clear();
    }
    for(int i = 0; i < objidxs.size(); i++) {
      Get().objidxs_.push_back(objidxs[i]);
    }
  } 
  inline static const std::vector<std::string> objidxs() { return Get().objidxs_; }
  
  // imgidxs
  inline static void set_imgidxs(const std::vector<std::string>& imgidxs) {
    if(!Get().imgidxs_.empty()) {
      Get().imgidxs_.clear();
    }
    for(int i = 0; i < imgidxs.size(); i++) {
      Get().imgidxs_.push_back(imgidxs[i]);
    }
  } 
  inline static const std::vector<std::string> imgidxs() { return Get().imgidxs_; }
  inline static void set_depth3(const cv::Mat& depth3ori){
    Get().depth3_ = depth3ori;
  }
  inline static const cv::Mat&  Depth3(){ return Get().depth3_; }

  // images_paths
  inline static void set_images_paths(const std::vector<std::string>& images_paths) {
    if(!Get().images_paths_.empty()) {
      Get().images_paths_.clear();
    }
    for(int i = 0; i < images_paths.size(); i++) {
      Get().images_paths_.push_back(images_paths[i]);
    }
  } 
  inline static const std::vector<std::string> images_paths() { return Get().images_paths_; }
  
  // stellate
  inline static void setSpiltCodeBoundWithStellate() {
      Get().stellate_ = "*******************************************************";
  }
  inline static std::string SpiltCodeBoundWithStellate() { return Get().stellate_; }
 
 protected:
  static shared_ptr<GlobalVars> singleton_;

  Phase phase_;
  int caffe_iter_;
  float learn_lr_;
  std::string stellate_;
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string>  images_paths_;
  cv::Mat depth3_;


 private:
  // The private constructor to avoid duplicate instantiation.
  GlobalVars();

  DISABLE_COPY_AND_ASSIGN(GlobalVars);
};

}  // namespace caffe

#endif  // CAFFE_UTIL_GLOBALVARS_HPP_
