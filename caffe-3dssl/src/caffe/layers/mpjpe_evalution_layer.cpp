#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mpjpe_evalution_layer.hpp"
#include "caffe/util/util_txt.hpp"

namespace caffe {

template <typename Dtype>
void MPJPEEvaluationLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sample_num_ = this->layer_param_.mpjpe_param().sample_num();
  scale_ = this->layer_param_.mpjpe_param().scale();
  const string save_filepath = this->layer_param_.mpjpe_param().save_filepath();
  save_file_.open(save_filepath.c_str());
  CHECK(save_file_.is_open() != false)  << "unable to open file for write result at " 
                                        << save_filepath;
  sample_ind_ = 0;
  error_ = Dtype(0.);

  const bool max_min_normalize = this->layer_param_.mpjpe_param().max_min_normalize();
  const string max_min_source = this->layer_param_.mpjpe_param().max_min_source();
  if (max_min_normalize != false){
    CHECK_GT(max_min_source.size(),0);
    load_txt(max_min_source, this->max_min_value_);
    CHECK_GE(this->max_min_value_.size(), 2);
    max_min_value_[0][0] += 0.000001;
    max_min_value_[0][1] += 0.000001;
    max_min_value_[0][2] += 0.000001;
    
  }else{
    this->max_min_value_.clear();
  }
  LossLayer<Dtype>::LayerSetUp(bottom,top);
}

template <typename Dtype>
void MPJPEEvaluationLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  one_mulplier_.Reshape(1,bottom[0]->channels(),1,1);
  caffe_gpu_set(one_mulplier_.count(), Dtype(1.0), one_mulplier_.mutable_gpu_data());
  frame_error_.Reshape(bottom[0]->num(),1,1,1);

}

// template <typename Dtype>
// void MPJPEEvaluationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   int count = bottom[0]->count();
//   caffe_sub(
//       count,
//       bottom[0]->cpu_data(),
//       bottom[1]->cpu_data(),
//       diff_.mutable_cpu_data());
//   Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
//   Dtype loss = dot / bottom[0]->num() / Dtype(2);
//   top[0]->mutable_cpu_data()[0] = loss;
// }


#ifdef CPU_ONLY
STUB_GPU(MPJPEEvaluationLayer);
#endif

INSTANTIATE_CLASS(MPJPEEvaluationLayer);
REGISTER_LAYER_CLASS(MPJPEEvaluation);

}  // namespace caffe
