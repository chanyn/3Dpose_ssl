#include <vector>

#include "caffe/my_layers/3dheatmap_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/* botoom[2] :2d heatmap
   botoom[0] :predict heatmap
   botoom[1] :labrl heatmap
*/

template <typename Dtype>
void HeatMap3DLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());

  diff_.ReshapeLike(*bottom[0]);
  weighted_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HeatMap3DLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* heatmap2d = bottom[2]->cpu_data();
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      weighted_diff_.mutable_cpu_data());

  const int batch_num = bottom[2]->num();
  const int key_point_num = bottom[2]->channels();
  for(int n = 0;n < batch_num;n++)
    for(int k = 0;k < key_point_num;k++) 
      for(int h = 0; h < bottom[2]->height(); h++)
        for(int w = 0; w < bottom[2]->width(); w++) {
          Dtype tmp = heatmap2d[bottom[2]->offset(n,k,h,w)];
          if(tmp == 0) {
            diff_.mutable_cpu_data()[diff_.offset(n,3*k,h,w)] = 0;
            diff_.mutable_cpu_data()[diff_.offset(n,3*k+1,h,w)] = 0;
            diff_.mutable_cpu_data()[diff_.offset(n,3*k+2,h,w)] = 0;

            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k,h,w)] = 0;
            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k+1,h,w)] = 0;
            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k+2,h,w)] = 0;
          }
          else {
            diff_.mutable_cpu_data()[diff_.offset(n,3*k,h,w)] *= tmp;
            diff_.mutable_cpu_data()[diff_.offset(n,3*k+1,h,w)] *= tmp;
            diff_.mutable_cpu_data()[diff_.offset(n,3*k+2,h,w)] *= tmp;

            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k,h,w)] *= tmp * tmp;
            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k+1,h,w)] *= tmp * tmp;
            weighted_diff_.mutable_cpu_data()[weighted_diff_.offset(n,3*k+2,h,w)] *= tmp * tmp;
          }
        }
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
  // LOG(INFO)<<"------------HHHHH-----------:"<<loss;
  // LOG(INFO)<<"------------HHHHH-----------:["<<diff_.cpu_data()<<"]-------------";
}

template <typename Dtype>
void HeatMap3DLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          weighted_diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HeatMap3DLossLayer);
#endif

INSTANTIATE_CLASS(HeatMap3DLossLayer);
REGISTER_LAYER_CLASS(HeatMap3DLoss);

}  // namespace caffe
