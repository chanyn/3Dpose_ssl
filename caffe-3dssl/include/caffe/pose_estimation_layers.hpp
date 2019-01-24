  // Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_POSE_ESTIMATION_LAYERS_HPP_
#define CAFFE_POSE_ESTIMATION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/my_layers/global_variables.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"




using std::vector;
using std::string;
namespace caffe {

/**
 * @brief Computes the accuracy for human pose/joint estimation
 * Using Percentage of Detected Joints (PDJ)
 * We restrict that the coordinates are in this interval [0, width - 1] and [0, height - 1].
 * So if use regression and in the beginning normalize coordinates, you must re-normalize them,
 * before use this layer, be calling `revert_normalized_pose_coords_layer` layer.
 *
 * Please refer to  `Deeppose: Human Pose Estimation via Deep Neural Networks, CVPR 2014.`
 */
template <typename Dtype>
class PosePDJAccuracyLayer : public Layer<Dtype> {
 public:
  
  explicit PosePDJAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PosePDJAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  /// PosePDJAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  void initAccFactors();
  void InitQuantization();
  void CalAccPerImage(const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr);
  void WriteResults(const float total_accuracies[]);
  void QuanFinalResults();
  void Quantization(const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr, const int num);

 protected:

  int label_num_;
  int images_num_;
  int key_point_num_;
  int images_itemid_;
  int acc_factor_num_;
  int shoulder_id_;
  int hip_id_;
  
  float acc_factor_;

  std::string acc_path_;
  std::string acc_name_;
  std::string acc_file_;
  std::string log_name_;
  std::string log_file_;

  std::vector<float> acc_factors_;
  std::vector< std::vector<float> > accuracies_;

  Blob<Dtype> diff_;
};

/**
 * @brief `heat map to coordinates` is that it creates coordinates from heat maps
 * simply find location of the maximum respondence in heat map and then according to
 * the mapping relationship between input image and the heat map, find the predicted
 * coordinate (x, y) of some part/joint
 *
 * Please refer to `Efficient Object Localization Using Convolutional Networks, CVPR 2014.`
 */
template <typename Dtype>
class HeatMapsFromCoordsLayer  : public Layer<Dtype> {
 public:
  explicit HeatMapsFromCoordsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "HeatMapsFromCoords"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // heat maps, mask, heat map info
  // virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 3; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void GetHeatMapHeightAndWidth(const vector<Blob<Dtype>*>& bottom);
  virtual void CreateHeatMapsFromCoords(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;

  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int max_width_;
  int max_height_;
  int img_width_;
  int img_height_;

  int batch_num_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;

  // float gaussian_sig_;
  // float gaussian_mean_;
  // heat_map_a_ * valid_dist_factor_
  float valid_dist_factor_;
  Blob<Dtype> prefetch_heat_maps_;  // 
  // Blob<Dtype> prefetch_heat_maps_masks_;  // 
};

template <typename Dtype>
class Gen3DHeatMapLayer  : public Layer<Dtype> {
 public:
  explicit Gen3DHeatMapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "Gen3DHeatMapLayer"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // heat maps, mask, heat map info
  // virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 3;   
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void GetHeatMapHeightAndWidth(const vector<Blob<Dtype>*>& bottom);
  virtual void CreateHeatMapsFromCoords(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int max_width_;
  int max_height_;
  int img_width_;
  int img_height_;
  int img_channel_;

  int batch_num_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;

  float valid_dist_factor_;
  Blob<Dtype> prefetch_heat_maps_;  
};

template <typename Dtype>
class Gen3DHeatMap2Layer  : public Layer<Dtype> {
 public:
  explicit Gen3DHeatMap2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "Gen3DHeatMap2Layer"; 
  }
  // labels, aux info
  // virtual inline int ExactNumBottomBlobs() const { return 2; }
  // heat maps, mask, heat map info
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }   
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CreateHeatMapsFromCoords(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int max_width_;
  int max_height_;
  int img_width_;
  int img_height_;
  int img_channel_;

  int batch_num_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;

  float valid_dist_factor_;
  Blob<Dtype> prefetch_heat_maps_; 
  Blob<Dtype> prefetch_3d_heat_maps_; 
};

template <typename Dtype>
class Gen3DXYZHeatMapLayer  : public Layer<Dtype> {
 public:
  explicit Gen3DXYZHeatMapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "Gen3DXYZHeatMapLayer"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // heat maps, mask, heat map info
  // virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 3;   
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void GetHeatMapHeightAndWidth(const vector<Blob<Dtype>*>& bottom);
  virtual void CreateHeatMapsFromCoords(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int max_width_;
  int max_height_;
  int img_width_;
  int img_height_;
  int img_channel_;

  int batch_num_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;

  float valid_dist_factor_;
  Blob<Dtype> prefetch_heat_maps_;  
};

/**
 * @brief `heat map to coordinates` is that it creates coordinates from heat maps
 * simply find location of the maximum respondence in heat map and then according to
 * the mapping relationship between input image and the heat map, find the predicted
 * coordinate (x, y) of some part/joint
 *
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 */
template <typename Dtype>
class CoordsFromHeatMapsLayer : public Layer<Dtype> {
 public:
  explicit CoordsFromHeatMapsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "CoordsFromHeatMaps"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CreateCoordsFromHeatMap(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  // virtual void FilterHeatMap(const vector<Blob<Dtype>*>& bottom);
  // virtual void FilterHeatMapWithLocationCon(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  // virtual void FilterHeatMapWithDepthValueCon(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  
  int batch_num_;
  int heat_channels_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;
  int per_heatmap_ch_;

  Blob<Dtype> prefetch_coordinates_scores_;  // 
  Blob<Dtype> prefetch_coordinates_labels_;  // default for coordinates
};

/**
 * @brief Computes the Euclidean (L2) loss @f$
 * bottom[0]: predicted heat maps
 * bottom[1]: ground truth heat maps
 * bottom[2]: ground truth masks (indicator for which heat maps are invalid)
 */
template <typename Dtype>
class PoseHeatMapLossLayer : public LossLayer<Dtype> {
 public:
  explicit PoseHeatMapLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "PoseHeatMapLoss"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * Unlike most loss layers, in the PoseHeatMapLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc PoseHeatMapLossLayer
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void PrintLoss();
  virtual void ComputesHeatMapLoss(const vector<Blob<Dtype>*>& bottom);
  virtual void CopyDiff(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @copydoc PoseHeatMapLossLayer
  /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom);

  void PrintLoss_gpu();
  void CheckRandNum_gpu();
  void ComputesHeatMapLoss_gpu(const vector<Blob<Dtype>*>& bottom);
  void CopyDiff_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/

  int heat_num_;
  int key_point_num_;
  int loss_emphase_type_;
  Dtype fg_eof_, bg_eof_, ratio_;
  
  /// when divided by UINT_MAX, 
  /// the randomly generated values @f$u\sim U(0,1)@f$
  unsigned int uint_thres_;
  
  Blob<Dtype> diff_;
  Blob<unsigned int> rand_vec_;
};

/**
 * @brief Normalize the coordinates
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 * bottom[1]: aux info (img_ind, ori-width, ori-height, im_scale)
 * top[0]: origin coordinates (need to be rescaled, that means x = x' / im_scale)
 */
template <typename Dtype>
class RescaledPoseCoordsLayer : public Layer<Dtype>{
 public:
  explicit RescaledPoseCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "RescaledPoseCoords"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
 protected:
  int num_;
  int channels_;
  int height_;
  int width_;
};


// if bottom.size() == 1:
//  bottom[0]: either predicted or ground truth
// if bottom.size() == 2:
//  bottom[0]: predicted
//  bottom[1]: ground truth
// Visualize Heat maps for both predicted and ground truth 
template <typename Dtype>
class VisualizedHeatMapsLayer : public Layer<Dtype>{
 public:
  explicit VisualizedHeatMapsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "VisualizedHeatMaps"; 
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void WriteFiles(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void WriteImages(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
  int num_;
  int channels_;
  int height_;
  int width_;
  int heat_num_;
  // 0: file-format, 1: image-format
  int visual_type_;
  Dtype threshold_;
  // path
  std::string heat_map_path_;
  std::string heat_map_files_name_;
  std::string heat_map_files_path_;
  std::string heat_map_images_name_;
  std::string heat_map_images_path_;
  // std::string phase_;
  // std::string gt_name_;
  // std::string pred_name_;
  // std::string fusion_name_;
  std::string phase_name_;
  std::string phase_path_;
  std::string img_ext_;
  std::string file_ext_;

  // come from caffe/util/global_variables.hpp
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
};

}  // namespace caffe

#endif  // CAFFE_POSE_ESTIMATION_LAYERS_HPP_