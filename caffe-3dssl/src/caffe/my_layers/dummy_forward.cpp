#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/my_layers/dummy_forward.hpp"

namespace caffe {

template <typename Dtype>
void DummyForwardLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // We create a parameter that is not used, and set param_propagate_down to true
  // This ensures that higher layers backpropagate to at least this layer.
  std::vector<int> dummy_shape { 1 };
  this->blobs_.resize( 1 );
  this->blobs_[0].reset(new Blob<Dtype>( dummy_shape ));

  this->param_propagate_down_.resize( 1, true );

}

template <typename Dtype>
void DummyForwardLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // Copy data
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i];
  }
}

template <typename Dtype>
void DummyForwardLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Copy diffs
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }
  }
}

INSTANTIATE_CLASS(DummyForwardLayer);
REGISTER_LAYER_CLASS(DummyForward);
}  // namespace caffe
