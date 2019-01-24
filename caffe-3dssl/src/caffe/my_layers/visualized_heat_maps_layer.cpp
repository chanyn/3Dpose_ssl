#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/pose_estimation_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/my_layers/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// if bottom.size() == 1:
//  bottom[0]: either predicted or ground truth
// if bottom.size() == 2:
//  bottom[0]: predicted
//  bottom[1]: ground truth

template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_visual_heat_maps_param());
  const VisualHeatMapsParameter visual_heat_maps_param =
      this->layer_param_.visual_heat_maps_param();
  CHECK(visual_heat_maps_param.has_heat_map_path());
  CHECK(visual_heat_maps_param.has_heat_map_files_name());
  CHECK(visual_heat_maps_param.has_heat_map_images_name());
  CHECK(visual_heat_maps_param.has_visual_type());
  CHECK(visual_heat_maps_param.has_threshold());
  CHECK(visual_heat_maps_param.has_phase_name());
  // CHECK(visual_heat_maps_param.has_phase());
  // CHECK(visual_heat_maps_param.has_gt_name());
  // CHECK(visual_heat_maps_param.has_pred_name());
  // CHECK(visual_heat_maps_param.has_fusion_name());
  this->heat_map_path_ = visual_heat_maps_param.heat_map_path();
  this->heat_map_files_name_ = visual_heat_maps_param.heat_map_files_name();
  this->heat_map_images_name_ = visual_heat_maps_param.heat_map_images_name();
  this->visual_type_ = visual_heat_maps_param.visual_type();
  this->threshold_ = Dtype(visual_heat_maps_param.threshold());
  this->phase_name_ = visual_heat_maps_param.phase_name();
  // use default values
  this->img_ext_ = visual_heat_maps_param.img_ext();
  this->file_ext_ = visual_heat_maps_param.file_ext();
  // this->phase_ = visual_heat_maps_param.phase();
  // this->gt_name_ = visual_heat_maps_param.gt_name();
  // this->pred_name_ = visual_heat_maps_param.pred_name();
  // this->fusion_name_ = visual_heat_maps_param.fusion_name();

  CHECK_GE(this->visual_type_, 0);
  CHECK_LE(this->visual_type_, 1);
  CHECK_GE(this->threshold_, Dtype(0.0));
  CHECK_LE(this->threshold_, Dtype(1.0));

  this->heat_map_files_path_ = this->heat_map_path_ + this->heat_map_files_name_;
  this->heat_map_images_path_ = this->heat_map_path_ + this->heat_map_images_name_;
  this->phase_path_ = this->visual_type_ == 0 ?
      this->heat_map_files_path_ + this->phase_name_ :
      this->heat_map_images_path_ + this->phase_name_;

  // CreateDir(this->heat_map_path_);
  // CreateDir(this->heat_map_files_path_);
  // CreateDir(this->heat_map_images_path_);
  // CreateDir(this->phase_path_);
}

template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  this->heat_num_ = this->height_ * this->width_;
  
  for(int idx = 0; idx < bottom.size(); idx++) {
    CHECK_EQ(this->num_, bottom[0]->num());
    CHECK_EQ(this->channels_, bottom[0]->channels());
    CHECK_EQ(this->height_, bottom[0]->height());
    CHECK_EQ(this->width_, bottom[0]->width());
  }
}

// if bottom.size() == 2, then use this order for bottom blobs: 
//    predicted, ground truth
template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::WriteFiles(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    // const std::string objidx = this->objidxs_[item_id];
    // const std::string imgidx = this->imgidxs_[item_id];

    for(int part_idx = 0; part_idx < this->channels_; part_idx++) {
      // get file handler
      const std::string file_path = this->phase_path_ + to_string(item_id) + "_"  + to_string(part_idx) + this->file_ext_;
      std::ofstream filer(file_path.c_str());
      CHECK(filer);

      // write results
      for(int bs = 0; bs < bottom.size(); bs++) {
        for(int h = 0; h < this->height_; ++h) {
          for(int w = 0; w < this->width_; ++w) {
            filer << bottom[bs]->data_at(item_id, part_idx, h, w) << " ";
          }
          filer << std::endl;
        }
        filer << std::endl;
        if(bs != bottom.size() - 1) {
          filer << GlobalVars::SpiltCodeBoundWithStellate();
          filer << std::endl;
        }
      }

      // close
      filer.close();
    }
  }
}

template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::WriteImages(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // variables
  // const int RGB = 3;
  const int thickness = 0.2;
  const Dtype One = Dtype(1);
  // const Dtype Zero = Dtype(0);
  const Dtype Depth = Dtype(255);
  const cv::Scalar color1(216, 16, 216);
  const cv::Scalar color2(16, 196, 21);
  // const cv::Scalar gt_color(255, 16, 21);
  // const cv::Scalar pred_color(21, 16, 255);
  std::vector<cv::Scalar> colors;
  colors.push_back(color1);
  colors.push_back(color2);
// __asm__("int $3");
  // deal with each image
  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    // const std::string objidx = this->objidxs_[item_id];
    // const std::string imgidx = this->imgidxs_[item_id];
    const std::string image_path = this->phase_path_ + to_string(item_id) + "_" + this->img_ext_;

    const int b_size = bottom.size();
    cv::Mat heat_map = cv::Mat::zeros(
        b_size * this->height_, this->channels_ * this->width_, CV_8UC3);
    // cv::Mat img = cv::Mat::zeros(this->height_, this->width_, CV_8UC3);
     for(int bs = 0; bs < b_size; bs++) {
      for(int c = 0; c < this->channels_; c++) {
        // convert each heat map into color image
        cv::Mat img = cv::Mat::zeros(this->height_, this->width_, CV_8UC3);
        for(int h = 0; h < this->height_; h++) {
          for(int w = 0; w < this->width_; w++) { 
            const Dtype v = bottom[bs]->data_at(item_id, c, h, w);
            uchar v2 = 0;
            if (v < this->threshold_) v2 = 0;
            else if (v > One) v2 = 255;
            else{
              // if (v !=0)
              //   LOG(INFO)<<"LLLL:"<<v<<":::"<<this->channels_;
              v2 = static_cast<uchar>(v * Depth);
            } 
            // set pixel
            img.at<cv::Vec3b>(h, w) = cv::Vec3b(v2, v2, v2);
            if (v2 == 255) {
            LOG(INFO) << "w " << w;
            LOG(INFO) << "h " << h;
            LOG(INFO) << this->threshold_;
            }
          }
        }  
        // copy img to heat_map
        // top_left.x, top_left.y, width, height
        cv::Rect rect(c * this->width_, bs * this->height_, this->width_, this->height_);
        cv::Mat rect_img = heat_map(rect) ;
        img.copyTo(rect_img);
      }
    }

    // draw line to distinguish each heat map for each joint/part 
    // and between predicted and ground truth
    for(int bs = 0; bs < b_size; bs++) {
      for(int idx = 1 ; idx < this->channels_; idx++) {
        const cv::Point p1(idx * this->width_, bs * this->height_ - 1);
        const cv::Point p2(idx * this->width_, (bs + 1) * this->height_ - 1);
        // img, p1, p2, color, thickness, lineType, shift
        cv::line(heat_map, p1, p2, colors[bs], thickness);
      }
    }
    // save
    LOG(INFO) << image_path;
    cv::imwrite(image_path, heat_map);
  }
}

template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // int heatmap_channels_ = visual_heat_maps_param.heatmap_channels();
  //  Blob<Dtype> *bottom_blob = bottom[i];
  // if( heatmap_channels_ > 1) {
  //   int kpn = bottom[0]->channels() / heatmap_channels_;
  //   for (int k = 0; k < kpn; k++)
  //     for(int x=0; x < )
  //     caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 64, 64, 64, (Dtype)1., F, X2_i, (Dtype)1., temp2);//diff_X1



  // }
  if(this->visual_type_ == 0) {
    this->WriteFiles(bottom, top);
  } else if(this->visual_type_ == 1) {
    this->WriteImages(bottom, top);
  } else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void VisualizedHeatMapsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(VisualizedHeatMapsLayer);
#endif

INSTANTIATE_CLASS(VisualizedHeatMapsLayer);
REGISTER_LAYER_CLASS(VisualizedHeatMaps);

}  // namespace caffe