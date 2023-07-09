#include "dataloader_objdet_2d.hpp"
#include <glog/logging.h>

namespace DataLoader{


DataLoaderObjDet2D::DataLoaderObjDet2D(){

}

int DataLoaderObjDet2D::get_batch(
            std::function<std::shared_ptr<cv::Mat>(const cv:: Mat&)> resize_strategy,
            std::function<int(float*, const cv::Mat&)> flatten_strategy,
            std::vector<float*> & data_ptrs){
    CHECK(data_ptrs.size() == 1);
    float * data_ptr = data_ptrs[0];
    const int batch_size = _image_buffer.size();

    while (!_image_buffer.empty()){
        const auto image_ptr = _image_buffer.front();
        _image_buffer.pop();
        const auto resized_image_ptr = resize_strategy(*image_ptr);
        // cv::imwrite("/data/binfeng/projects/server_multi-platform/images/bus.jpg_resize_result.jpg", *resized_image_ptr);
        // LOG(INFO) << "finished resize! size : " << resized_image_ptr->rows << " " << resized_image_ptr->cols;
        data_ptr += flatten_strategy(data_ptr, *resized_image_ptr);
    }
    return batch_size;
}



bool DataLoaderObjDet2D::push(const std::shared_ptr<cv::Mat> &image){
    if (_image_buffer.size() == MAX_BATCH_SIZE){
        _image_buffer.pop();
    }
    _image_buffer.push(image);

    return true;
}


} // DataLoader


