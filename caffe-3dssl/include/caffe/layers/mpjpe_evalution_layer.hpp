#ifndef MPJPE_EVALUATIOM_LAYER_HPP
#define MPJPE_EVALUATIOM_LAYER_HPP

#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MPJPEEvaluationLayer : public LossLayer<Dtype> {
public:
  explicit MPJPEEvaluationLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), diff_() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MPJPEEvaluation"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  
  // if bottom blos == 1 will be a L1 norm term

  // virtual inline int MinBottomBlobs() const { return 1; }
  // virtual inline int MaxBottomBlobs() const { return 3; }

  /**
  * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  * to both inputs -- override to return true and always allow force_backward.
  */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return false;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    NOT_IMPLEMENTED;
  };
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };
  void Project_to_origin(Blob<Dtype> *predict_blob);
  void unnormalize(const vector<Blob<Dtype> *> &bottom);
  int sample_num_;
  int sample_ind_;
  Dtype error_;
  Blob<Dtype> diff_;
  Blob<Dtype> one_mulplier_;
  Blob<Dtype> frame_error_;

  Dtype scale_;
  std::ofstream save_file_;

  vector<vector<float> > max_min_value_;
};

}  // namespace caffe

#endif  // D3_POSE_LAYER_HPP
