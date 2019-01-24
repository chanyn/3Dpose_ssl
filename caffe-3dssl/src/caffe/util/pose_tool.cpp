// Copyright 2015 ddk

#include <stdint.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h> 
#include <dirent.h>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;

namespace caffe {

int string_to_int(const std::string str) {
  int num = 0;
  for(int i = 0; i < str.length(); i++) {
    num = num * 10 + (str[i] - '0');
  }

  return num;
}

// normal integer
std::string to_string(int number) {
   std::string number_string = "";
   char ones_char = '0';
   int ones = 0;
   while(true){
     ones = number % 10;
     switch(ones){
         case 0: ones_char = '0'; break;
         case 1: ones_char = '1'; break;
         case 2: ones_char = '2'; break;
         case 3: ones_char = '3'; break;
         case 4: ones_char = '4'; break;
         case 5: ones_char = '5'; break;
         case 6: ones_char = '6'; break;
         case 7: ones_char = '7'; break;
         case 8: ones_char = '8'; break;
         case 9: ones_char = '9'; break;
         default : break;
     }
     number -= ones;
     number_string = ones_char + number_string;
     if(number == 0){
         break;
     }
     number = number/10;
   }
   return number_string;
}

/*
  idx: [0, 2, 4, ..., label_num - 2]
*/
template <typename Dtype>
void NormalizationPoseCoordinates(
    Dtype& width, Dtype& height, const int idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables) 
{
  if(normalized_type == 1) {
    width -= (real_width / 2);
    width /= (real_width + 0.0);

    height -= (real_height / 2);
    height /= (real_height + 0.0);

  } else if(normalized_type == 2) {
    width -= coordinate_averages[idx];
    width /= (coordinate_standard_variables[idx] + 0.);

    height -= coordinate_averages[idx + 1];
    height /= (coordinate_standard_variables[idx + 1] + 0.);
  // ohters
  } else {
    // check default
    if(normalized_type != 0) {
      LOG(INFO) << "invalid normalized_type: " 
          << normalized_type;
      // SUBJECTIVE_EXIT;
    }
  }
}
template void NormalizationPoseCoordinates<float>(
    float& width, float& height, const int idx,
    const float real_width, const float real_height,
    const int normalized_type,
    std::vector<float> coordinate_averages, 
    std::vector<float> coordinate_standard_variables);
template void NormalizationPoseCoordinates<double>(
    double& width, double& height, const int idx,
    const double real_width, const double real_height,
    const int normalized_type,
    const std::vector<double> coordinate_averages, 
    const std::vector<double> coordinate_standard_variables);

/*
  idx: [0, 2, 4, ..., label_num - 2]
*/
template <typename Dtype>
void NormalizationPoseCoordinates(
    Dtype& s_width, Dtype& s_height,
    Dtype& e_width, Dtype& e_height,
    const int s_idx, const int e_idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables) 
{
  NormalizationPoseCoordinates(s_width, 
      s_height, s_idx, real_width, real_height, normalized_type, 
      coordinate_averages, coordinate_standard_variables);
  NormalizationPoseCoordinates(e_width, 
      e_height, e_idx, real_width, real_height, normalized_type, 
      coordinate_averages, coordinate_standard_variables);
}
template void NormalizationPoseCoordinates<float>(
    float& s_width, float& s_height,
    float& e_width, float& e_height,
    const int s_idx, const int e_idx,
    const float real_width, const float real_height,
    const int normalized_type,
    std::vector<float> coordinate_averages, 
    std::vector<float> coordinate_standard_variables);
template void NormalizationPoseCoordinates<double>(
    double& s_width, double& s_height,
    double& e_width, double& e_height,
    const int s_idx, const int e_idx,
    const double real_width, const double real_height,
    const int normalized_type,
    std::vector<double> coordinate_averages, 
    std::vector<double> coordinate_standard_variables);

/*
  idx: [0, 2, 4, ..., label_num - 2]
*/
template <typename Dtype>
void ReverseNormalizationPoseCoordinates(
    Dtype& width, Dtype& height, const int idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables) 
{ 
  if(normalized_type == 0) {
    return;
  }
  if(normalized_type == 1) {
    width *= real_width;
    width += (real_width / 2);

    height *= real_height;
    height += (real_height / 2);

  } else if(normalized_type == 2) {
    width *= (coordinate_standard_variables[idx] + 0.);
    width += coordinate_averages[idx];

    height *= (coordinate_standard_variables[idx + 1] + 0.);
    height += coordinate_averages[idx + 1];
  // ohters
  } else {
    // check default
    LOG(INFO) << "invalid normalized_type: " 
        << normalized_type;
    // SUBJECTIVE_EXIT;
  }
}
template void ReverseNormalizationPoseCoordinates<float>(
    float& width, float& height, const int idx,
    const float real_width, const float real_height,
    const int normalized_type,
    std::vector<float> coordinate_averages, 
    std::vector<float> coordinate_standard_variables);
template void ReverseNormalizationPoseCoordinates<double>(
    double& width, double& height, const int idx,
    const double real_width, const double real_height,
    const int normalized_type,
    const std::vector<double> coordinate_averages, 
    const std::vector<double> coordinate_standard_variables);


/*
  idx: [0, 2, 4, ..., label_num - 2]
*/
template <typename Dtype>
void ReverseNormalizationPoseCoordinates(
    Dtype& s_width, Dtype& s_height,
    Dtype& e_width, Dtype& e_height,
    const int s_idx, const int e_idx,
    const Dtype real_width, const Dtype real_height,
    const int normalized_type,
    const std::vector<Dtype> coordinate_averages, 
    const std::vector<Dtype> coordinate_standard_variables) 
{
  ReverseNormalizationPoseCoordinates(e_width,
      s_height, s_idx, real_width, real_height, normalized_type, 
      coordinate_averages, coordinate_standard_variables);
  ReverseNormalizationPoseCoordinates(e_width,
      e_height, e_idx, real_width, real_height, normalized_type, 
      coordinate_averages, coordinate_standard_variables);
}
template void ReverseNormalizationPoseCoordinates<float>(
    float& s_width, float& s_height,
    float& e_width, float& e_height,
    const int s_idx, const int e_idx,
    const float real_width, const float real_height,
    const int normalized_type,
    std::vector<float> coordinate_averages, 
    std::vector<float> coordinate_standard_variables);
template void ReverseNormalizationPoseCoordinates<double>(
    double& s_width, double& s_height,
    double& e_width, double& e_height,
    const int s_idx, const int e_idx,
    const double real_width, const double real_height,
    const int normalized_type,
    std::vector<double> coordinate_averages, 
    std::vector<double> coordinate_standard_variables);

template <typename Dtype>
void find_max_and_min_vals(const Dtype* data_ptr, const int low, 
  const int high, Dtype& max_val, Dtype& min_val) 
{
  if(low == high) {
    max_val = data_ptr[low];
    min_val = data_ptr[low];
  } else if(high - low == 1) {
    max_val = std::max(data_ptr[low], data_ptr[high]);
    min_val = std::min(data_ptr[low], data_ptr[high]);
  } else {
    Dtype low_max_val, low_min_val;
    Dtype high_max_val, high_min_val;

    // get middle index
    const int mid = (low + high) >> 1;

    find_max_and_min_vals(data_ptr, low, mid, low_max_val, low_min_val);
    find_max_and_min_vals(data_ptr, mid + 1, 
            high, high_max_val, high_min_val);

    max_val = std::max(low_max_val, high_max_val);
    min_val = std::min(low_min_val, high_min_val);
  }
} 
// Explicit instantiation
template void find_max_and_min_vals<int>(
    const int* data_ptr, const int low, const int high, 
    int& max_val, int& min_val);
template void find_max_and_min_vals<float>(
    const float* data_ptr, const int low, const int high, 
    float& max_val, float& min_val);
template void find_max_and_min_vals<double>(
    const double* data_ptr, const int low, const int high, 
    double& max_val, double& min_val);

template <typename Dtype>
void find_max_and_min_vals(
    const Dtype* data_ptr, 
    const int low, const int high, 
    Dtype& max_val, int& max_val_idx, 
    Dtype& min_val, int& min_val_idx) 
{
  if(low == high) {
    min_val = data_ptr[low];
    max_val = data_ptr[high];
    min_val_idx = low;
    max_val_idx = high;
  } else if(high - low == 1) {
    if(data_ptr[low] > data_ptr[high]) {
      min_val = data_ptr[high];
      max_val = data_ptr[low];
      min_val_idx = high;
      max_val_idx = low;
    } else {
      min_val = data_ptr[low];
      max_val = data_ptr[high];
      min_val_idx = low;
      max_val_idx = high;
    }
  } else {
    // min
    Dtype low_max_val, low_min_val;
    int low_min_val_idx, low_max_val_idx;
    // max
    Dtype high_max_val, high_min_val;
    int high_min_val_idx, high_max_val_idx;

    // get middle index
    const int mid = (low + high) >> 1;

    find_max_and_min_vals(data_ptr, low, mid, 
        low_max_val, low_max_val_idx, 
        low_min_val, low_min_val_idx);
    find_max_and_min_vals(data_ptr, mid + 1, high, 
        high_max_val, high_max_val_idx, 
        high_min_val, high_min_val_idx);

    // max
    if(low_max_val > high_max_val) {
      max_val = low_max_val;
      max_val_idx = low_max_val_idx;
    } else {
      max_val = high_max_val;
      max_val_idx = high_max_val_idx;
    }

    // min
    if(low_min_val > high_min_val) {
      min_val = high_min_val;
      min_val_idx = high_min_val_idx;
    } else {
      min_val = low_min_val;
      min_val_idx = low_min_val_idx;
    }
  }
}
// Explicit instantiation
template void find_max_and_min_vals<int>(
    const int* data_ptr, const int low, 
    const int high, int& max_val, int& max_val_idx, 
    int& min_val, int& min_val_idx);
template void find_max_and_min_vals<float>(
    const float* data_ptr, const int low, 
    const int high, float& max_val, int& max_val_idx, 
    float& min_val, int& min_val_idx);
template void find_max_and_min_vals<double>(
    const double* data_ptr, const int low, 
    const int high, double& max_val, int& max_val_idx, 
    double& min_val, int& min_val_idx);

// divide an number into two number, l
// ike num = n1 * n2, w.r.t min|n1 - n2|
void get_two_factors(int number, int& first, int& second) {
  int sqrt_val = (int)(std::sqrt(number) + 0.5);

  for(int sv = sqrt_val; sv > 0; sv++) {
    if(number % sv == 0){
      first = sv;
      second = number / sv;
      break;
    }
  }
} 

void MkdirTree(string sub, string dir){
  if (sub.length() == 0)
    return;

  int i=0;
  for (; i<sub.length(); i++){
    dir += sub[i];
    if (sub[i] == '/')
        break;
  }
  // build (sub) root directory
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // recur
  if (i+1 < sub.length()){
    MkdirTree(sub.substr(i+1), dir);
  }
}

bool IsDirectory(std::string dir) {
  struct stat st;
  if(stat(dir.c_str(), &st) == 0){
    if((st.st_mode & S_IFMT) == S_IFDIR){
      return true;
    }
  }

  return false;
}

bool DirectoryExists(const char* pzPath) {
  if (pzPath == NULL) {
    return false;
  }

  DIR *pDir;
  bool bExists = false;
  pDir = opendir (pzPath);

  if(pDir != NULL) {
    bExists = true;    
    (void) closedir (pDir);
  }

  return bExists;
}

bool CreateDir(const std::string sPathName) {
  bool flag = false;
  if(!DirectoryExists(sPathName.c_str())) {
    mkdir(sPathName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  flag = true;
  return flag;
}

int CreateDir(const char *sPathName, int beg) {
  char DirName[256];
  strcpy(DirName, sPathName);
  int i, len = strlen(DirName);
  if (DirName[len - 1] != '/')
    strcat(DirName, "/");

  len = strlen(DirName);

  for (i = beg; i < len; i++) {
    if (DirName[i] == '/') {
      DirName[i] = 0;
      if (access(DirName, F_OK) != 0) {
        if (mkdir(DirName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) 
        {
          LOG(ERROR)<< "Failed to create folder " 
              << sPathName;
        }
      }
      DirName[i] = '/';
    }
  }

  return 0;
}

// index starts from zero
void get_skeleton_idxs(const std::string source,
    std::vector<int>& start_skel_idxs, std::vector<int>& end_skel_idxs) {
  std::ifstream filer(source.c_str());
  CHECK(filer);
  LOG(INFO) << "skeleton file path: " << source;

  int skel_num;
  int start_idx, end_idx;

  filer >> skel_num;
  LOG(INFO) << "number of skeleton: " << skel_num;

  for(int idx = 0; idx < skel_num; idx++) {
    filer >> start_idx >> end_idx;
    start_skel_idxs.push_back(start_idx);
    end_skel_idxs.push_back(end_idx);
    LOG(INFO) << "idx: " << idx << ", start_idx: " << start_idx << ", end_idx: " << end_idx;
  }
}

// index starts from zero
const std::vector<std::vector<int> > get_skeleton_idxs(const std::string source) {
  std::ifstream filer(source.c_str());
  CHECK(filer);
  LOG(INFO) << "skeleton file path: " << source;

  int skel_num;
  int start_idx, end_idx;
  std::vector<int> end_skel_idxs;
  std::vector<int> start_skel_idxs;
  std::vector<std::vector<int> >skel_idxs;

  filer >> skel_num;
  LOG(INFO) << "number of skeleton: " << skel_num;

  for(int idx = 0; idx < skel_num; idx++) {
    filer >> start_idx >> end_idx;
    start_skel_idxs.push_back(start_idx);
    end_skel_idxs.push_back(end_idx);
    LOG(INFO) << "idx: " << idx << ", start_idx: " << start_idx << ", end_idx: " << end_idx;
  }

  skel_idxs.push_back(start_skel_idxs);
  skel_idxs.push_back(end_skel_idxs);
  return skel_idxs;
}

} // namespace caffe