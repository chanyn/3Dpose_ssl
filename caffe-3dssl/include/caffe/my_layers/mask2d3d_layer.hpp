#ifndef MASK2D3D_HPP
#define MASK2D3D_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype> 
 class Mask2D3DLayer : public Layer<Dtype> {
public:
  explicit Mask2D3DLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Mask"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  // virtual inline int MinTopBlobs() const { return 1; }

  /**
  * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  * to both inputs -- override to return true and always allow force_backward.
  */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return false;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int skel_num;
};

}  // namespace caffe

#endif 
