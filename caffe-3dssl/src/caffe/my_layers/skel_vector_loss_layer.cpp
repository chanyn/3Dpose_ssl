#include <vector>

#include "caffe/my_layers/skelvector_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe {

template <typename Dtype>
void SkelVectorLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  bone_1.ReshapeLike(*bottom[0]);
  bone_2.ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void SkelVectorLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{  
  LossLayer<Dtype> ::LayerSetUp(bottom,top);
  CHECK(this->layer_param_.has_skel_vector_param());
  this->dim = this->layer_param_.skel_vector_param().dim();
}

template <typename Dtype>
void SkelVectorLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  const int num = bottom[0] -> num();
  const int num_skel = bottom[0] -> channels() / this->dim; // for 3d pose
  // int parent_node[] = {0, 0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15 };
  int parent_node[] = {0, 0,1,2, 0,4,5, 0,7,8, 8,10,11, 8,13,14}; 
  const Dtype *blob_data1 = bottom[0] -> cpu_data();
  const Dtype *blob_data2 = bottom[1] -> cpu_data();
  Dtype *bone_data1 = bone_1.mutable_cpu_data();
  Dtype *bone_data2 = bone_2.mutable_cpu_data();
  
  for(int n = 0; n < num; ++n) {
    // except the root node
    for(int c = 0; c < num_skel; ++c){
      for (int d = 0; d < this->dim; ++d) {
        bone_data1[bottom[0]->offset(n,this->dim*c+d)] = blob_data1[bottom[0]->offset(n,this->dim*parent_node[c]+d)] - blob_data1[bottom[0]->offset(n,this->dim*c+d)];
        bone_data2[bottom[1]->offset(n,this->dim*c+d)] = blob_data2[bottom[1]->offset(n,this->dim*parent_node[c]+d)] - blob_data2[bottom[1]->offset(n,this->dim*c+d)];
      }
    //   bone_data1[bottom[0]->offset(n,3*c)] = blob_data1[bottom[0]->offset(n,3*parent_node[c])] - blob_data1[bottom[0]->offset(n,3*c)];
    //   bone_data1[bottom[0]->offset(n,3*c+1)] = blob_data1[bottom[0]->offset(n,3*parent_node[c]+1)] - blob_data1[bottom[0]->offset(n,3*c)+1];
    //   bone_data1[bottom[0]->offset(n,3*c+2)] = blob_data1[bottom[0]->offset(n,3*parent_node[c])+2] - blob_data1[bottom[0]->offset(n,3*c)+2];

    //   bone_data2[bottom[1]->offset(n,3*c)] = blob_data2[bottom[1]->offset(n,3*parent_node[c])] - blob_data2[bottom[1]->offset(n,3*c)];
    //   bone_data2[bottom[1]->offset(n,3*c+1)] = blob_data2[bottom[1]->offset(n,3*parent_node[c]+1)] - blob_data2[bottom[1]->offset(n,3*c)+1];
    //   bone_data2[bottom[1]->offset(n,3*c+2)] = blob_data2[bottom[1]->offset(n,3*parent_node[c])+2] - blob_data2[bottom[1]->offset(n,3*c)+2];
    }
  }
  caffe_sub(
      count,
      bone_1.cpu_data(),
      bone_2.cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SkelVectorLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? -1 : 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SkelVectorLossLayer);
#endif

INSTANTIATE_CLASS(SkelVectorLossLayer);
REGISTER_LAYER_CLASS(SkelVectorLoss);

}  // namespace caffe
