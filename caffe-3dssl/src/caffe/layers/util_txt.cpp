#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include "boost/algorithm/string.hpp"
#include "caffe/util/util_txt.hpp"

namespace caffe {


  /*
   * input format 
   *    1,2,3,4
   *    5,6,7,8
   */
void load_txt(const std::string max_min_source, std::vector<std::vector<float> > &max_min_value){
  std::ifstream infile(max_min_source.c_str());
  std::string value_str;

  max_min_value.clear();
  while(infile >> value_str){
    std::vector<std::string> value_info;
    boost::trim(value_str);
    boost::split(value_info, value_str, boost::is_any_of(","));
    int num_label = value_info.size();
    std::vector<float> value_vec(num_label,0);
    for (int i = 0; i < num_label; ++i){
      float value = atof(value_info[i].c_str());
      value_vec[i] = value;
    }
    max_min_value.push_back(value_vec);
  }
}


}  // namespace caffe

