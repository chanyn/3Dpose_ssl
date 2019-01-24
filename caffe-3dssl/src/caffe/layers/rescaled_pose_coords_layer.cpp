#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
// #include "caffe/vision_layers.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void RescaledPoseCoordsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  CHECK_EQ(this->channels_ % 2, 0);
  CHECK_EQ(this->channels_, bottom[0]->count() / this->num_);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
}

template <typename Dtype>
void RescaledPoseCoordsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  CHECK_EQ(this->channels_, bottom[0]->count() / this->num_);
  CHECK_EQ(this->channels_ % 2, 0);

  // aux info (img_ind, width, height, im_scale, flippable)
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->channels(), 5);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  // reshape
  top[0]->Reshape(this->num_, this->channels_, this->height_, this->width_);
  CHECK_EQ(bottom[0]->count(), top[0]->count());
}

template <typename Dtype>
void RescaledPoseCoordsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bootom_coords = bottom[0]->cpu_data();
  const Dtype* aux_info = bottom[1]->cpu_data();
  Dtype* top_coords = top[0]->mutable_cpu_data();

  int offset = 0;
  for(int item_id = 0; item_id < this->num_; item_id++) {
    // (img_ind, width, height, im_scale, flippable)
    const int aux_info_offset = bottom[1]->offset(item_id);
    // const Dtype img_ind     = aux_info[aux_info_offset + 0];
    // const Dtype img_width   = aux_info[aux_info_offset + 1];
    // const Dtype img_height  = aux_info[aux_info_offset + 2];
    const Dtype im_scale    = aux_info[aux_info_offset + 3];
    // const Dtype flippable   = aux_info[aux_info_offset + 4];
    // LOG(INFO) << "aux_info_offset: " << aux_info_offset << ", im_scale: " << im_scale;

    for(int idx = 0; idx < this->channels_; idx++) {
      top_coords[offset] = bootom_coords[offset] / im_scale;
      offset++;
    }
  }
  CHECK_EQ(offset, bottom[0]->count());
  CHECK_EQ(offset, top[0]->count());
}

template <typename Dtype>
void RescaledPoseCoordsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RescaledPoseCoordsLayer);
#endif

INSTANTIATE_CLASS(RescaledPoseCoordsLayer);
REGISTER_LAYER_CLASS(RescaledPoseCoords);

}  // namespace caffe