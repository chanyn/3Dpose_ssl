#ifndef CAFFE_UTIL_CONVERT_IMG_BLOB_H_
#define CAFFE_UTIL_CONVERT_IMG_BLOB_H_

#include <unistd.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using cv::Point_;
using cv::Mat_;
using cv::Mat;
using cv::vector;

namespace caffe {

cv::Mat ImageRead(const string& filename, const bool is_color = true);

}  // namespace caffe

#endif   // CAFFE_UTIL_CONVERT_IMG_BLOB_H_
