// Copyright 2015 DDK

#include <cmath>
#include <cstdlib>
#include <cstring>
// CUDA's, not caffe's, for fabs, signbit
#include <math_functions.h>  
#include <thrust/device_vector.h>
// thrust::plus
#include <thrust/functional.h>  
#include <thrust/reduce.h>

#include "caffe/util/pose_tool.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void caffe_find_max_and_min_vals_kernel(
		const int loop_n, const Dtype* data_ptr, 
    Dtype& max_val, int& max_val_idx, 
		Dtype& min_val, int& min_val_idx) {
  CUDA_KERNEL_LOOP(idx, loop_n) {
    if(max_val < data_ptr[idx]) {
      max_val = data_ptr[idx];
      max_val_idx = idx;
    }
    if(min_val > data_ptr[idx]) {
      min_val = data_ptr[idx];
      min_val_idx = idx;
    }
  }
}

template <>
void find_max_and_min_vals_gpu(
    const int N, const int* data_ptr, 
    int& max_val, int& max_val_idx, 
		int& min_val, int& min_val_idx) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_find_max_and_min_vals_kernel<int>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, data_ptr, max_val, max_val_idx, min_val, min_val_idx);
}

template <>
void find_max_and_min_vals_gpu(
    const int N, const float* data_ptr, 
    float& max_val, int& max_val_idx, 
    float& min_val, int& min_val_idx) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_find_max_and_min_vals_kernel<float>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, data_ptr, max_val, max_val_idx, min_val, min_val_idx);
}

template <>
void find_max_and_min_vals_gpu(
    const int N, const double* data_ptr, 
    double& max_val, int& max_val_idx, 
    double& min_val, int& min_val_idx) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_find_max_and_min_vals_kernel<double>
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, data_ptr, max_val, max_val_idx, min_val, min_val_idx);
}

}  // namespace caffe
