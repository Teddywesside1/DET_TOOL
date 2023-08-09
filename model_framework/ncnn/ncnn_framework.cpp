#include "ncnn_framework.hpp"


#include <fstream>
#include <iostream>
#include <chrono>
#include "util_path.hpp"
#include <glog/logging.h>
#include <glog/log_severity.h>


namespace ModelFramework{
namespace NCNN{

int NCNNModelFramework::maxBatch = 4;
int NCNNModelFramework::maxHeight = 640;
int NCNNModelFramework::maxWidth = 640;
int NCNNModelFramework::maxChannel = 3;


NCNNModelFramework::NCNNModelFramework(const std::string model_param_path, 
                                        const std::string model_bin_path)
                                    : _ncnn_ex(_ncnn_model.create_extractor()){
    loadNCNNModel(model_param_path, model_bin_path);
    prepareBuffer();
}


void NCNNModelFramework::loadNCNNModel(const std::string model_param_path,
                                        const std::string model_bin_path){
    // _ncnn_model.opt.use_vulkan_compute = true;
    if (_ncnn_model.load_param(model_param_path.c_str()) == -1
    || _ncnn_model.load_model(model_bin_path.c_str()) == -1){
        LOG(ERROR) << "failed to load NCNN Model! \
                        param_path : " << model_param_path 
                    << "\tbin_path : " << model_bin_path; 
        throw std::runtime_error("could not load ncnn model!");
    }

    _ncnn_ex = _ncnn_model.create_extractor();
}


void NCNNModelFramework::prepareBuffer(){
    const int total_blobs_count = _ncnn_model.input_indexes().size() 
                                    + _ncnn_model.output_indexes().size();
    _buffer.resize(total_blobs_count);
    for (int i = 0 ; i < _ncnn_model.input_indexes().size() ; ++ i){
        _buffer[i] = new u_char[maxBatch * maxHeight * maxWidth * maxChannel];
    }

    _output_blob_buffer.resize(_ncnn_model.output_indexes().size());
}

void NCNNModelFramework::framework_forward(const int batch_size,
                                            const int input_height,
                                            const int input_width,
                                            const int input_channel){
    const int input_blobs_total = _ncnn_model.input_indexes().size();
    const auto & input_blobs_names = _ncnn_model.input_names();
    std::vector<ncnn::Mat> inputs(input_blobs_total);
    for (int i = 0 ; i < input_blobs_total ; ++ i){
        inputs.emplace_back(input_width, input_height, input_channel, _buffer[i]);
        _ncnn_ex.input(input_blobs_names[i], inputs[i]);
    }

    const auto & output_blobs_names = _ncnn_model.output_names();
    for (int i = 0 ; i < output_blobs_names.size() ; ++ i){
        LOG(INFO) << "output i : " << i << "\tname : " << output_blobs_names[i]
                    << "output_blob_buffer size : " << _output_blob_buffer.size();
        _ncnn_ex.extract(output_blobs_names[i], _output_blob_buffer[i]);
        _buffer[input_blobs_total + i] = _output_blob_buffer[i];
    }
    
}

std::vector<void*>& NCNNModelFramework::get_buffer(){
    return _buffer;
}


NCNNModelFramework::~NCNNModelFramework(){
    for (void * ptr : _buffer){
        if (ptr != nullptr){
            free(ptr);
        }
    }
    _buffer.clear();
}

} // NCNN

} // ModelFramework