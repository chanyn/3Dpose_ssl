#include <vector>

#include "caffe/my_layers/mask2d3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Mask2D3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int channel = bottom[0]->count() / bottom[0]->shape(0);
  this->skel_num = channel / 3;
  srand((unsigned)time(NULL)); 
  CHECK_EQ(top.size(), 2);
}

template <typename Dtype>
void Mask2D3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[1]->ReshapeLike(*bottom[0]);
  top[0]->Reshape(bottom[0]->shape(0), this->skel_num * 2,1,1);
}

template <typename Dtype>
void Mask2D3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  // const int count = top[0]->count();
  Dtype* top_data2 = top[0]->mutable_cpu_data(); 
  Dtype* top_data3 = top[1]->mutable_cpu_data(); 
  // generate random mask_flag
  for (int n = 0; n < bottom[0]->num(); n++)
  	for (int k = 0; k < this->skel_num; k++) {
  		Dtype mask_flag = rand() / double(RAND_MAX);
  		if (k == 7 || k == 9) {
  			if (mask_flag > 0.4) {
          top_data2[top[0]->offset(n,2 * k)] = 1;
  			  top_data2[top[0]->offset(n,2 * k + 1)] = 1; 

          top_data3[top[1]->offset(n,3 * k)] = 1;
          top_data3[top[1]->offset(n,3 * k + 1)] = 1; 
          top_data3[top[1]->offset(n,3 * k + 2)] = 1; 
  			}
  			else {
          top_data2[top[0]->offset(n,2 * k)] = 0;
          top_data2[top[0]->offset(n,2 * k + 1)] = 0; 
          
          top_data3[top[1]->offset(n,3 * k)] = 0;
          top_data3[top[1]->offset(n,3 * k + 1)] = 0; 
          top_data3[top[1]->offset(n,3 * k + 2)] = 0;
        }
  		}
  		else {
  			if (mask_flag > 0.2) {
          top_data2[top[0]->offset(n,2 * k)] = 1;
          top_data2[top[0]->offset(n,2 * k + 1)] = 1; 

          top_data3[top[1]->offset(n,3 * k)] = 1;
          top_data3[top[1]->offset(n,3 * k + 1)] = 1; 
          top_data3[top[1]->offset(n,3 * k + 2)] = 1; 
        }
        else {
          top_data2[top[0]->offset(n,2 * k)] = 0;
          top_data2[top[0]->offset(n,2 * k + 1)] = 0; 
          
          top_data3[top[1]->offset(n,3 * k)] = 0;
          top_data3[top[1]->offset(n,3 * k + 1)] = 0; 
          top_data3[top[1]->offset(n,3 * k + 2)] = 0;
        }
  		}
  	}
}

template <typename Dtype>
void Mask2D3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i)
      bottom_diff[i] = 0;
  }
}

#ifdef CPU_ONLY
STUB_GPU(Mask2D3DLayer);
#endif

INSTANTIATE_CLASS(Mask2D3DLayer);
REGISTER_LAYER_CLASS(Mask2D3D);
}  // namespace caffe
