#ifndef INVERTED_ORDER_HPP
#define INVERTED_ORDER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
 class InvertedOrderLayer : public NeuronLayer<Dtype>{
public:
  explicit InvertedOrderLayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InvertedOrder"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // virtual inline int MinBottomBlobs() const { return 1; }
  // virtual inline int MaxBottomBlobs() const { return 2; }
  
  // if bottom blos == 1 will be a L1 norm term

  // virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  //   virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 2; }
  

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
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      //const vector<Blob<Dtype>*>& top){
    //NOT_IMPLEMENTED;
  //};
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif 
