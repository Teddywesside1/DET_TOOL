#pragma once
#include "CMakeConfig.h"
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace DataLoader{

class IDataLoader{
public:
    /**
     * @brief Get the batch object
     * 
     * @param flatten_strategy 
     * @param data_ptrs 
     * @return int 
     */
    virtual int
    get_batch(std::function<std::shared_ptr<cv::Mat>(const cv:: Mat&)> resize_strategy,
            std::function<int(float*, const cv::Mat&)> flatten_strategy,
            std::vector<float*> & data_ptrs) = 0;


    virtual bool
    push(const std::shared_ptr<cv::Mat> & image)= 0;
};


class DataLoaderFactoryBase{
public:
    virtual IDataLoader* createDataLoader() = 0;
};    

} // DataLoader

