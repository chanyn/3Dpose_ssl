#ifndef CAFFE_HEATMAP_HPP_
#define CAFFE_HEATMAP_HPP_

#include <string>
#include <vector>
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{


template<typename Dtype>
class DataHeatmapLayer: public BasePrefetchingDataLayer<Dtype>
{

public:

    explicit DataHeatmapLayer(const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~DataHeatmapLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "DataHeatmap"; }

    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }


protected:
    virtual void load_batch(Batch<Dtype>* batch);
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void ShuffleImages();

    // Global vars
    shared_ptr<Caffe::RNG> rng_data_;
 
    int cur_img_;    
    string root_img_dir_;
    // vector of (image, label) pairs
    vector< pair<string, pair<vector<float>, vector<float> > > > img_label_list_;    
};

}

#endif /* CAFFE_HEATMAP_HPP_ */