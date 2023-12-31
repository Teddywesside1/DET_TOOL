#pragma once
#include "CMakeConfig.h"
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "interface_process_strategy.hpp"

namespace DataLoader{

struct BatchInfo{
    int origin_height;
    int origin_width;
    float scale;
};

class IDataLoader{
public:
    /**
     * @brief Get the batch object
     * 
     * @param resize_strategy 
     * @param flatten_strategy 
     * @param data_ptrs 
     * @param batch_info 
     * @return int 
     */
    virtual int
    get_buffer_as_one_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info) = 0;
    
    virtual bool
    get_explicit_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info,
            const int batch_size) = 0;


    virtual bool
    push(const std::shared_ptr<cv::Mat> & image)= 0;
};


class DataLoaderFactoryBase{
public:
    virtual IDataLoader* createDataLoader() = 0;
};    

} // DataLoader

