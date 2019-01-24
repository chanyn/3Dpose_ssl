#ifndef CAFFE_UTIL_UTIL_IMG_H_
#define CAFFE_UTIL_UTIL_IMG_H_

#include <vector>
#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using cv::Point_;
using cv::Mat_;
using cv::Mat;
using cv::vector;

#define PI 3.14159265

namespace caffe {

template<typename Dtype>
Mat_<Dtype> Get_Affine_matrix(
		const Point_<Dtype>& srcCenter, 
		const Point_<Dtype>& dstCenter, 
		const Dtype alpha,
    const Dtype scale);

template<typename Dtype>
Mat_<Dtype> inverseMatrix(const Mat_<Dtype>& M);

/*
 * fill_type specifies how to fill the pixels output of original image
 * if fill_type is equal true, then fill the pixel with value
 * else use the nearest pixel the in the original image to fill it
 */
template<typename Dtype>
void mAffineWarp(const Mat_<Dtype>& M, const Mat& srcImg,
	  Mat& dstImg, const bool fill_type = true, 
	  const uchar value = 0);

/*cv::Mat BlobToGreyImage(const Blob<Dtype>* blob, const int n, const int c, 
		const Dtype scale = Dtype(1.0));*/

template <typename Dtype>
cv::Mat BlobToColorImage(const Blob<Dtype>* blob, const int n);

template <typename Dtype>
cv::Mat BlobToColorImage(const Blob<Dtype>* blob, const int n,
		const std::vector<Dtype> mean_values);

template <typename Dtype>
cv::Mat BlobTooneDImage(const Blob<Dtype>* blob, const int n);

template <typename Dtype>
void ImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image);

template <typename Dtype>
void ImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image,
		const std::vector<Dtype> mean_values);

template <typename Dtype>
void oneDImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image);

template <typename Dtype>
void ResizeBlob_cpu(
		const Blob<Dtype>* src, 
		const int src_n, 
		const int src_c,
		Blob<Dtype>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff = true 
		/* true: data, false: diff */
);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_Data_cpu(const Blob<Dtype>* src, 
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_Diff_cpu(const Blob<Dtype>* src, 
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_cpu(
		const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, 
		Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(
		const Blob<Dtype>* src, Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, 
		Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src, 
	const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, 
		const int dst_c, 
		/* true: data, false: diff */
		const bool data_or_diff = true);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst, 
		/* true: data, false: diff */
		const bool data_or_diff = true);

template <typename Dtype>
void ResizeBlob_Data_gpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_Diff_gpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst);

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, 
	const int src_h, const int src_w,
		Dtype* dst, const int dst_h, const int dst_w);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(const Dtype* src,Dtype* dst, 
		const int dst_h, const int dst_w,
		const Dtype* loc1, const Dtype* weight1, 
		const Dtype* loc2,const Dtype* weight2,
		const	Dtype* loc3,const Dtype* weight3,
		const Dtype* loc4, const Dtype* weight4);

template <typename Dtype>
void GetBiLinearResizeMatRules_cpu(
		const int src_h, const int src_w,
		 const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, 
		Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, 
		Dtype* loc4, Dtype* weight4);

template <typename Dtype>
void GetBiLinearResizeMatRules_gpu(
		const int src_h, const int src_w,
		 const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, 
		Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, 
		Dtype* loc4, Dtype* weight4);

void CropAndResizePatch(
		const cv::Mat& src, cv::Mat& dst, 
		const vector<float>& coords,
		const cv::Size& resize_size, 
		const bool is_fill = true, 
		const int fill_value = 0);

template <typename Dtype>
void GetResizeRules(const int src_height, const int src_width,
		const int dst_height, const int dst_width,
		Blob<Dtype>* weights, Blob<int>* locs,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count = 1, const int num = 1);

template <typename Dtype>
void AffineWarpBlob_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count = 1, const int num = 1);

template <typename Dtype>
void AffineWarpBlob_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const Blob<Dtype>* weights, const Blob<int>* locs);

template <typename Dtype>
void AffineWarpBlob_gpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count = 1, const int num = 1);
template <typename Dtype>
void AffineWarpBlob_gpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const Blob<Dtype>* weights, const Blob<int>* locs);

template<typename Dtype>
void CropAndResizeBlob(const Blob<Dtype>& src, Blob<Dtype>& dst, const vector<float>& coords,
		const bool is_fill = true, const Dtype fill_value = Dtype(0.0));

}  // namespace caffe

#endif   // CAFFE_UTIL_UTIL_IMG_H_
