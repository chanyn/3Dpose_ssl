#include <vector>

#include "caffe/my_layers/mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.has_mask_param());
  this->dim = this->layer_param_.mask_param().dim();
  this->prob = this->layer_param_.mask_param().prob();
  int channel = bottom[0]->count() / bottom[0]->shape(0);
  this->skel_num = channel / this->dim;
  srand((unsigned)time(NULL));  
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  // const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data(); 
  // generate random mask_flag
  for (int n = 0; n < bottom[0]->num(); n++)
  	for (int k = 0; k < this->skel_num; k++) {
  		Dtype mask_flag = rand() / double(RAND_MAX);
  		if (k == 7 || k == 9) {
  			if (mask_flag > (2*this->prob)) {
  				for (int d = 0; d < this->dim; d++)
  					top_data[top[0]->offset(n,this->dim * k + d)] = 1;
  				// top_data[top[0]->offset(n,this->dim * k)] = 1;
  			 //    top_data[top[0]->offset(n,this->dim * k + 1)] = 1; 
  			}
  			else 
  				for (int d = 0; d < this->dim; d++)
  					top_data[top[0]->offset(n,this->dim * k + d)] = 0;
  		}
  		else {
  			if (mask_flag > this->prob) 
  				for (int d = 0; d < this->dim; d++)
  					top_data[top[0]->offset(n,this->dim * k + d)] = 1;
  			else 
  				for (int d = 0; d < this->dim; d++)
  					top_data[top[0]->offset(n,this->dim * k + d)] = 0;
  		}
  	}
  // for (int n = 0;n < top[0]->num(); n++)
  // 	for (int c = 0;c < top[0]->channels(); c++)
  // 		LOG(INFO) << top_data[top[0]->offset(n,c)];
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    // const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i)
      bottom_diff[i] = 0;
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);
}  // namespace caffe
