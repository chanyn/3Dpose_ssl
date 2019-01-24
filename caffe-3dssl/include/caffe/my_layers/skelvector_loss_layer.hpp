#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SkelVectorLossLayer : public LossLayer<Dtype> {
 public:
  explicit SkelVectorLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SkelVectorLoss"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //   const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // void Gen_Bone(Blob<Dtype> *p_blob, Blob<Dtype> *bone_blob);

  Blob<Dtype> diff_;
  Blob<Dtype> bone_1;
  Blob<Dtype> bone_2;
  int dim;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
