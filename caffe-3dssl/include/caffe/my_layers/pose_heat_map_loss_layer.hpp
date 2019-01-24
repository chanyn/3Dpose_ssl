#ifndef POSE_HEAT_MAP_LOSS_LAYER_HPP
#define POSE_HEAT_MAP_LOSS_LAYER_HPP

#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
//#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/common.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/my_layers/global_variables.hpp"


using std::vector;
using std::string;


namespace caffe{

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

} // namespace caffe
#endif /* POSE_HEAT_MAP_LOSS_LAYER_HPP */
