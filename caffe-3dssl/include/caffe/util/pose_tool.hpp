#ifndef CAFFE_SMALL_TOOL_HPP_
#define CAFFE_SMALL_TOOL_HPP_

#include <vector>
#include <string>
#include "boost/scoped_ptr.hpp"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

namespace caffe {

int string_to_int(const std::string str);

std::string to_string(int number);

void get_two_factors(int number, int& first, int& second);

template <typename Dtype>
void NormalizationPoseCoordinates(
    Dtype& s_width, Dtype& s_height,
    Dtype& e_width, Dtype& e_height,
    const int s_idx, const int e_idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables);

template <typename Dtype>
void NormalizationPoseCoordinates(
    Dtype& width, Dtype& height, const int idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables) ;

template <typename Dtype>
void ReverseNormalizationPoseCoordinates(
    Dtype& width, Dtype& height, const int idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    std::vector<Dtype> coordinate_averages, 
    std::vector<Dtype> coordinate_standard_variables);

template <typename Dtype>
void ReverseNormalizationPoseCoordinates(
    Dtype& s_width, Dtype& s_height,
    Dtype& e_width, Dtype& e_height,
    const int s_idx, const int e_idx);

template <typename Dtype>
void find_max_and_min_vals(
		const Dtype* data_ptr, 
		const int low, const int high, 
		Dtype& max_val, Dtype& min_val);

template <typename Dtype>
void find_max_and_min_vals(
		const Dtype* data_ptr, 
		const int low, const int high, 
		Dtype& max_val, int& max_val_idx, 
		Dtype& min_val, int& min_val_idx);

template <typename Dtype>
void find_max_and_min_vals_gpu(
		const Dtype* data_ptr, 
		const int low, const int high, 
		Dtype& max_val, Dtype& min_val);

template <typename Dtype>
void find_max_and_min_vals_gpu(
		const int N, 
		const Dtype* data_ptr, 
		Dtype& max_val, int& max_val_idx, 
		Dtype& min_val, int& min_val_idx);

template <typename Dtype>
void find_max_and_min_vals_gpu(
		const Dtype* data_ptr, 
		const int low, const int high, 
		Dtype& max_val, int& max_val_idx, 
		Dtype& min_val, int& min_val_idx)
{
	find_max_and_min_vals_gpu(
		high - low + 1,
		data_ptr,
		max_val, max_val_idx,
		min_val, min_val_idx
	);
}

void MkdirTree(string sub, string dir);

bool IsDiretory(std::string dir);

bool DirectoryExists(const char* pzPath);

inline int CreateDir(const std::string sPathName, int beg) {
	if(beg <= 0) beg = 0;
	
	return CreateDir(sPathName.c_str(), beg);
}

int CreateDir(const char *sPathName, int beg = 0);

bool CreateDir(const std::string sPathName);

// index starts from zero
void get_skeleton_idxs(const std::string source,
    std::vector<int>& start_skel_idxs, std::vector<int>& end_skel_idxs);
const std::vector<std::vector<int> > get_skeleton_idxs(const std::string source);

}  // namespace caffe

#endif   // CAFFE_SMALL_TOOL_HPP_