#ifndef FMATRIX_LOSS_LAYER_HPP_
#define FMATRIX_LOSS_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
// #include <fstream>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class FmatrixLossLayer : public LossLayer<Dtype> {
public:
   explicit FmatrixLossLayer(const LayerParameter& param)
     : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FmatrixLossLayer"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  
  // // if bottom blos == 1 will be a L1 norm term

  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  // *
  // * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  // * to both inputs -- override to return true and always allow force_backward.
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //   const vector<Blob<Dtype>*>& top){
  //  // NOT_IMPLEMENTED;
  // };

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // void Project_to_origin(Blob<Dtype> *predict_blob);
  // void unnormalize(const vector<Blob<Dtype> *> &bottom);

  Dtype* product;//[17] ;
  Dtype* leftTerm;//[17 * 3] ;
  Dtype* rightTerm;//[17 * 3] ;
  // vector<Dtype>& product;
  Dtype X1[17][3],X2[17][3];
  Dtype F[9] ; 
  std::ofstream save_file_1;
  std::ofstream save_file_2;
  int sample_ind_;
  Blob<Dtype> xx1;
  Blob<Dtype> xx2;

};

}  // namespace caffe

#endif
