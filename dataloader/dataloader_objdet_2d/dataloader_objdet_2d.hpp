#pragma once

#include "interface_dataloader.hpp"
#include <queue>
#include "defs.hpp"

namespace DataLoader{

class DataLoaderObjDet2D : public IDataLoader{
public:
    DataLoaderObjDet2D();

    int get_batch(std::function<std::shared_ptr<cv::Mat>(const cv:: Mat&)> resize_strategy,
            std::function<int(float*, const cv::Mat&)> flatten_strategy,
            std::vector<float*> & data_ptrs) override;

    bool push(const std::shared_ptr<cv::Mat> & image) override;


private:
    std::queue<std::shared_ptr<cv::Mat>> _image_buffer;
};


} // DataLoader


