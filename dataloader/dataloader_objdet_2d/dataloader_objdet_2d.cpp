#include "dataloader_objdet_2d.hpp"
#include <glog/logging.h>

namespace DataLoader{


DataLoaderObjDet2D::DataLoaderObjDet2D(){

}

int DataLoaderObjDet2D::get_buffer_as_one_batch(
            ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info){
    CHECK(data_ptrs.size() == 1);
    float * data_ptr = data_ptrs[0];
    const int batch_size = _image_buffer.size();

    batch_info.clear();

    for (int i = 0 ; i < batch_size ; ++ i){
        const auto image_ptr = _image_buffer.front();
        _image_buffer.pop();
        cv::Mat resized_image;
        struct BatchInfo info;
        info.origin_height = image_ptr->rows;
        info.origin_width = image_ptr->cols;
        info.scale = resize_strategy(*image_ptr, resized_image);
        // cv::imwrite("/data/binfeng/projects/server_multi-platform/images/bus.jpg_resize_result.jpg", *resized_image_ptr);
        // LOG(INFO) << "finished resize! size : " << resized_image_ptr->rows << " " << resized_image_ptr->cols;
        data_ptr += flatten_strategy(data_ptr, resized_image);

        batch_info.push_back(info);
    }
    return batch_size;
}


bool
DataLoaderObjDet2D::get_explicit_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
        ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
        std::vector<float*> & data_ptrs,
        std::vector<BatchInfo> & batch_info,
        const int batch_size){
    CHECK(data_ptrs.size() == 1);
    if (_image_buffer.size() < batch_size) return false;

    float * data_ptr = data_ptrs[0];
    batch_info.clear();

    for (int i = 0 ; i < batch_size ; ++ i){
        const auto image_ptr = _image_buffer.front();
        _image_buffer.pop();
        cv::Mat resized_image;
        struct BatchInfo info;
        info.origin_height = image_ptr->rows;
        info.origin_width = image_ptr->cols;
        info.scale = resize_strategy(*image_ptr, resized_image);
        // cv::imwrite("/data/binfeng/projects/server_multi-platform/images/bus.jpg_resize_result.jpg", *resized_image_ptr);
        // LOG(INFO) << "finished resize! size : " << resized_image_ptr->rows << " " << resized_image_ptr->cols;
        data_ptr += flatten_strategy(data_ptr, resized_image);

        batch_info.push_back(info);
    }
    return true;
}


bool DataLoaderObjDet2D::push(const std::shared_ptr<cv::Mat> &image){
    if (_image_buffer.size() == MAX_BATCH_SIZE){
        _image_buffer.pop();
    }
    _image_buffer.push(image);

    return true;
}







BlockDataLoaderObjDet2D::BlockDataLoaderObjDet2D(const int maxBlockQueueSize)
                                                : _image_buffer(maxBlockQueueSize){

}

int BlockDataLoaderObjDet2D::get_buffer_as_one_batch(
            ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info){
    CHECK(data_ptrs.size() == 1);
    float * data_ptr = data_ptrs[0];
    const int batch_size = _image_buffer.size();

    batch_info.clear();
    int i = 0;
    for ( ; i < batch_size ; ++ i){
        const auto op_image_ptr = _image_buffer.pop();
        if (!op_image_ptr.has_value()){
            break;
        }
        const auto image_ptr = op_image_ptr.value();
        cv::Mat resized_image;
        struct BatchInfo info;
        info.origin_height = image_ptr->rows;
        info.origin_width = image_ptr->cols;
        info.scale = resize_strategy(*image_ptr, resized_image);
        // cv::imwrite("/data/binfeng/projects/server_multi-platform/images/bus.jpg_resize_result.jpg", *resized_image_ptr);
        // LOG(INFO) << "finished resize! size : " << resized_image_ptr->rows << " " << resized_image_ptr->cols;
        data_ptr += flatten_strategy(data_ptr, resized_image);

        batch_info.push_back(info);
    }
    return i;
}


bool
BlockDataLoaderObjDet2D::get_explicit_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
        ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
        std::vector<float*> & data_ptrs,
        std::vector<BatchInfo> & batch_info,
        const int batch_size){
    CHECK(data_ptrs.size() == 1);

    float * data_ptr = data_ptrs[0];
    batch_info.clear();

    for (int i = 0 ; i < batch_size ; ++ i){
        const auto op_image_ptr = _image_buffer.pop();
        if (!op_image_ptr.has_value()) return false;

        const auto image_ptr = op_image_ptr.value();
        cv::Mat resized_image;
        struct BatchInfo info;
        info.origin_height = image_ptr->rows;
        info.origin_width = image_ptr->cols;
        info.scale = resize_strategy(*image_ptr, resized_image);
        data_ptr += flatten_strategy(data_ptr, resized_image);

        batch_info.push_back(info);
    }
    return true;
}



bool BlockDataLoaderObjDet2D::push(const std::shared_ptr<cv::Mat> &image){
    _image_buffer.push(image);
    return true;
}


} // DataLoader


