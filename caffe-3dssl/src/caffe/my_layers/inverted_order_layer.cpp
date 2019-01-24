#include <algorithm>
#include <vector>
#include <string>
#include "caffe/my_layers/inverted_order_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
void InvertedOrderLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  // top blob
  top[0]->ReshapeLike(*bottom[0]);
  // top[1]->Reshape(bottom[0]->num(),1,1,1);

}

template <typename Dtype>
void InvertedOrderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
//   sample_ind_ = 0;
//   const string save_filepath1 = this->layer_param_.save_path_param().save_path();
//   save_file_1.open(save_filepath1.c_str());
//   CHECK(save_file_1.is_open() != false)  << "unable to open file for write result at " 
                                        // << save_filepath1;
  }

template <typename Dtype>
void InvertedOrderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_tem = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int n=0;n < bottom[0]->num();n++)
    for(int c=0; c < bottom[0]->channels();c++){
      top_data[top[0]->offset(bottom[0]->num() - n -1,c)] = bottom_tem[bottom[0]->offset(n,c)];
    }
  if(top.size() > 1)
     caffe_gpu_set(top[1]->count(),static_cast<Dtype>(1.0),top[1]->mutable_cpu_data());
  
}

template <typename Dtype>
void InvertedOrderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for(int n=0;n < bottom[i]->num();n++)
        for(int c=0; c < bottom[i]->channels();c++)
          bottom_diff[bottom[i]->offset(bottom[i]->num() - n -1,c)] = top_diff[top[0]->offset(n,c)];
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(InvertedOrderLayer);
#endif

INSTANTIATE_CLASS(InvertedOrderLayer);
REGISTER_LAYER_CLASS(InvertedOrder);

}  // namespace caffe
