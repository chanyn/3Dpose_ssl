// Copyright 2015 DDK (dongdk.sysu@foxmail.com)

#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>
#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

using std::min;
using std::max;
inline double round(double x){
  return floor(x + 0.5);
}
template <typename Dtype>
void Gen3DHeatMap2Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_gen_3dheatmap2_param());
  const Gen3DHeatMap2Parameter& gen_3dheatmap2_param = 
      this->layer_param_.gen_3dheatmap2_param();

  CHECK(gen_3dheatmap2_param.has_heat_map_a());
  CHECK(gen_3dheatmap2_param.has_heat_map_b());
  CHECK(gen_3dheatmap2_param.has_img_width());
  CHECK(gen_3dheatmap2_param.has_img_height());

  this->heat_map_a_  = gen_3dheatmap2_param.heat_map_a();
  this->heat_map_b_  = gen_3dheatmap2_param.heat_map_b();
  this->valid_dist_factor_  = gen_3dheatmap2_param.valid_dist_factor();
  this->img_width_ = gen_3dheatmap2_param.img_width();
  this->img_height_ = gen_3dheatmap2_param.img_height();

  LOG(INFO) << "*************************************************";
  LOG(INFO) << "is_binary: " << this->is_binary_;
  LOG(INFO) << "heat_map_a: " << this->heat_map_a_;
  LOG(INFO) << "heat_map_b: " << this->heat_map_b_;
  LOG(INFO) << "valid_dist_factor: " << this->valid_dist_factor_;
  LOG(INFO) << "**************************************************";
}


template <typename Dtype>
void Gen3DHeatMap2Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  this->batch_num_ = bottom[0]->num();
  this->label_num_ = bottom[0]->count() / this->batch_num_;
  this->key_point_num_ = this->label_num_ / 2;
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2);
  CHECK_EQ(this->label_num_, bottom[0]->channels());
  // CHECK_EQ(this->key_point_num_, bottom[1]->channels() / 3);
  // CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  const int batch_num_ = bottom[0]->num();
  int label_num_ = bottom[0]->count() / batch_num_;
  const int key_point_num_ = label_num_ / 2;
  CHECK_EQ(label_num_, bottom[0]->channels());

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  Dtype max_width = this->img_width_;
  Dtype max_height = this->img_height_;

  // reshape & reset
  const int mod_w = int(max_width) % this->heat_map_a_;
  if(mod_w > 0) max_width = max_width - mod_w + this->heat_map_a_;
  const int mod_h = int(max_height) % this->heat_map_a_;
  if(mod_h > 0) max_height = max_height - mod_h + this->heat_map_a_;
  
  /// I don't know whether has heat_map_b_ is good or not, 
  /// It looks like a translated factor...
  // CHECK_EQ(this->heat_map_b_, 0) << "here we don't need heat_map_b_, just set it to be 0...";
  this->max_width_ = max_width;
  this->max_height_ = max_height;
  this->heat_width_ = (this->max_width_ - this->heat_map_b_) / this->heat_map_a_;
  this->heat_height_ = (this->max_height_ - this->heat_map_b_) / this->heat_map_a_;
  this->heat_num_ = this->heat_width_ * this->heat_height_;

  // prefetch_coordinates_labels
  this->prefetch_heat_maps_.Reshape(
      bottom[0]->num(), 
      key_point_num_, 
      this->heat_height_, 
      this->heat_width_
  );

  // top blob
  top[0]->Reshape(bottom[0]->num(), key_point_num_, this->heat_height_, this->heat_width_);
  if(bottom.size() > 1) {
    CHECK_EQ(this->key_point_num_, bottom[1]->channels() / 3);
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    top[1]->Reshape(bottom[0]->num(), key_point_num_ * 3, this->heat_height_, this->heat_width_);
    this->prefetch_3d_heat_maps_.Reshape(
      bottom[0]->num(), 
      key_point_num_ * 3, 
      this->heat_height_, 
      this->heat_width_
    );
  }

}

template <typename Dtype> 
void Gen3DHeatMap2Layer<Dtype>::CreateHeatMapsFromCoords(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const Dtype Zero = Dtype(0);
  const Dtype* coords_ptr = bottom[0]->cpu_data();
  const Dtype* coords_3d_ptr = bottom[1]->cpu_data();
  Dtype* heat_maps_ptr = this->prefetch_heat_maps_.mutable_cpu_data();
  Dtype* heat_maps_3d_ptr = this->prefetch_3d_heat_maps_.mutable_cpu_data();
  caffe_set(
      this->prefetch_heat_maps_.count(), 
      Zero, 
      this->prefetch_heat_maps_.mutable_cpu_data()
  );
  
  caffe_set(
      this->prefetch_3d_heat_maps_.count(), 
      Zero, 
      this->prefetch_3d_heat_maps_.mutable_cpu_data()
  );
  int kpn_count;
  int coords_offset;
  // int heat_map_masks_offset;
  // control the affected area
  const int valid_len = round(this->heat_map_a_ * this->valid_dist_factor_);
  const Dtype valid_square = valid_len * valid_len; 
  // start
  for(int bn = 0; bn < batch_num_; bn++) {
    // get offset of coordinates
    coords_offset = bottom[0]->offset(bn);
    for(int kpn = 0; kpn < this->key_point_num_; kpn++) {
      // std::cout << this->key_point_num_ << std::endl;
      kpn_count = 0;
      const int idx = kpn * 2;
      // origin coordinates for one part (x, y)
      const Dtype x_3d = coords_3d_ptr[coords_offset + kpn * 3];
      const Dtype y_3d = coords_3d_ptr[coords_offset + kpn * 3 + 1];
      const Dtype z_3d = coords_3d_ptr[coords_offset + kpn * 3 + 2];
      const Dtype ox = coords_ptr[coords_offset + idx];
      const Dtype oy = coords_ptr[coords_offset + idx + 1];
      // mapping: heat map's coordinates for one part (hx, hy)
      const Dtype hx = round(ox * Dtype(this->heat_width_));
      const Dtype hy = round(oy * Dtype(this->heat_height_));

      // coordinates around (hx, hy)
      for(int oj = -valid_len; oj <= valid_len; oj++) {
        const Dtype hy2 = hy + oj;
        if(hy2 < 0 || hy2 >= this->heat_height_) { continue; }

        for(int oi = -valid_len; oi <= valid_len; oi++) {
          const Dtype hx2 = hx + oi;
          if(hx2 < 0 || hx2 >= this->heat_width_) { continue; }
          const Dtype square = oi * oi + oj * oj;
          if(square > valid_square) { continue; }
          kpn_count++;
          // get offset for heat maps
          const int h_idx = this->prefetch_heat_maps_.offset(bn, kpn, int(hy2), int(hx2));

            /*
              hx1 = (ox - a) / b, hy1 = (oy - a) / b
              dx = hx2 * a + b - ox = (hx1 + oi) * a + b - ox = (hx1 * a + b - ox) + oi * a ~~ oi * a
              dy = hy2 * a + b - oy = (hy1 + oj) * a + b - oy = (hy1 * a + b - oy) + oj * a ~~ oj * a
              oi ~ [-valid_len, valid_len]
              oj ~ [-valid_len, valid_len]
              means:
                heat map会损失coordinates的精度,因为原图到heat map的转换，heat map到原图的转换,
                会导致coordinates有着0.5*scale*heat_map_a的偏差(这里将heat_map_b设为0)
              dist in [1/(e^2), 1]
            */
          const Dtype dist = exp(- 2 * square / valid_square);
          // float tem = max(heat_maps_ptr[h_idx], dist);
          heat_maps_ptr[h_idx] =  max(heat_maps_ptr[h_idx], dist);
          heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3*kpn, int(hy2), int(hx2))] = x_3d * max(heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3 * kpn, int(hy2), int(hx2))], dist);
          heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3*kpn+1, int(hy2), int(hx2))] = y_3d * max(heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3 * kpn +1, int(hy2), int(hx2))], dist);
          heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3*kpn+2, int(hy2), int(hx2))] = z_3d * max(heat_maps_3d_ptr[this->prefetch_3d_heat_maps_.offset(bn, 3 * kpn + 2, int(hy2), int(hx2))], dist);
          // LOG(INFO)<<"channel: "<<kpn<< " : "<<tem;
        }
      }
    }
  }
}

template <typename Dtype>
void Gen3DHeatMap2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Get Heat maps for batch
   // __asm__("int $3");
  CreateHeatMapsFromCoords(bottom, top);
  // Copy the heat maps
  caffe_copy(
      top[0]->count(), 
      this->prefetch_heat_maps_.cpu_data(), 
      top[0]->mutable_cpu_data()
  );
  caffe_copy(
      top[1]->count(), 
      this->prefetch_3d_heat_maps_.cpu_data(), 
      top[1]->mutable_cpu_data()
  );
}

template <typename Dtype>
void Gen3DHeatMap2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU( Layer);
#endif

INSTANTIATE_CLASS(Gen3DHeatMap2Layer);
REGISTER_LAYER_CLASS(Gen3DHeatMap2);

}  // namespace caffe