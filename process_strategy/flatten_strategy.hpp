#pragma once

#include "interface_process_strategy.hpp"

namespace ModelInference{
namespace ProcessStrategy{

class FlattenStrategyHWC2CHW : public IFlattenStrategy{
public:
    int operator()(float* data_ptr, const cv::Mat& image);
};


class FlattenStrategyHWC2CHW_OpenMP : public IFlattenStrategy{
public:
    int operator()(float* data_ptr, const cv::Mat& image);
};

} // ProcessStrategy
}// ModelInference