#ifndef CAFFE_UTIL_TXT_
#define CAFFE_UTIL_TXT_

#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include "boost/algorithm/string.hpp"

namespace caffe {

  /*
   * input format 
   *    1,2,3,4
   *    5,6,7,8
   */
void load_txt(const std::string max_min_source, std::vector<std::vector<float> > &max_min_value);


}  // namespace caffe

#endif
