#pragma once

#include "interface_dataloader.hpp"
#include <queue>
#include "defs.hpp"

namespace DataLoader{

class DataLoaderObjDet2D : public IDataLoader{
public:
    DataLoaderObjDet2D();

    int get_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info) override;

    bool push(const std::shared_ptr<cv::Mat> & image) override;


private:
    std::queue<std::shared_ptr<cv::Mat>> _image_buffer;
};


} // DataLoader


