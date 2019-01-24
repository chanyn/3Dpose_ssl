#include <algorithm>
#include <vector>
#include <string>

#include "caffe/my_layers/save_path.hpp"

namespace caffe {


template <typename Dtype>
void SavePathLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

}

template <typename Dtype>
void SavePathLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  sample_ind_ = 0;
  const string save_filepath1 = this->layer_param_.save_path_param().save_path();
  save_file_1.open(save_filepath1.c_str());
  CHECK(save_file_1.is_open() != false)  << "unable to open file for write result at " 
                                        << save_filepath1;

  // const string save_filepath2 = this->layer_param_.save_path_param().save_path2();
  // save_file_2.open(save_filepath2.c_str());
  // CHECK(save_file_2.is_open() != false)  << "unable to open file for write result at " 
  //                                       << save_filepath;
  }

template <typename Dtype>
void SavePathLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // // unnormalize
  // for(int i = 0;i < bottom.size(); ++i){
    // Blob<Dtype> *bottom_blob = bottom[0];
    // const Dtype* bottom_data = bottom_blob -> cpu_data();
    // for(int n = 0; n < bottom_blob-> num(); ++n){
    //   for(int c =0; c < bottom_blob -> channels(); ++c) {
    //     LOG(INFO)<<bottom_blob[bottom[0]->offset(n,c)];
    //   }
    // }
  // } 

  // convert to orignal position
  const Dtype* xx1 = bottom[0]->cpu_data();
  Dtype x1[bottom[0]->num() * bottom[0]->channels()];

  if(bottom.size() > 1) {
    // const Dtype* bbox1 = bottom[1]->cpu_data();
 // for (int item = 0; item < bottom[0]->num(); ++item)
  // for (int c=0; c < bottom[0]->channels();c+=2) {
  //   x1[bottom[0]->offset(item,c)] = xx1[bottom[0]->offset(item,c)] + bbox1[bottom[1]->offset(item,0)];
  //   x1[bottom[0]->offset(item,c+1)] = xx1[bottom[0]->offset(item,c+1)] + bbox1[bottom[1]->offset(item,1)];
  //  }
//  // for (int i=0; i < bottom[0]->channels();i++) {
 // //   x1[i] = xx1[i] + bbox1[0];
 // //   x1[i+1] = xx1[i+1] + bbox1[1];
  ////  }
LOG(INFO)<<"!!!!!!!!!!!!!!!!!!!";
    const Dtype* im_scale = bottom[1]->cpu_data();
    for (int item = 0; item < bottom[0]->num(); ++item)
      for (int c = 0; c < bottom[0]->channels(); c++)
        x1[bottom[0]->offset(item,c)] = xx1[bottom[0]->offset(item,c)] / im_scale[bottom[1]->offset(item)];
  /////////////////////// save //////////////////////////
  for(int ind = 0; ind < bottom[0] -> num(); ++ind) {
    // save_file_1 << ind + offset + 1;
    // save_file_2 << ind + offset + 1;
    for(int c = 0; c < bottom[0] -> channels(); ++c){
        save_file_1 << "," << x1[bottom[0] -> offset(ind,c)];
    }
    save_file_1 << std::endl;
  }
  }
  else {
    for(int ind = 0; ind < bottom[0] -> num(); ++ind) {
      for(int c = 0; c < bottom[0] -> channels(); ++c){
        save_file_1 << "," << xx1[bottom[0] -> offset(ind,c)];
    }
    save_file_1 << std::endl;
  }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SavePathLayer);
#endif

INSTANTIATE_CLASS(SavePathLayer);
REGISTER_LAYER_CLASS(SavePath);

}  // namespace caffe
