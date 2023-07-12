#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

namespace ModelInference{
namespace ProcessStrategy{

class IResizeStrategy{
public:
    virtual float operator()(const cv::Mat& src_image,
                            cv::Mat& dst_image) = 0;
protected:
    IResizeStrategy(const int target_height, const int target_width)
                : _target_height(target_height)
                , _target_width(target_width) {}

    const int _target_height;
    const int _target_width;
};


class IFlattenStrategy{
public:
    virtual int operator()(float* data_ptr, const cv::Mat& image) = 0;
};

} // ProcessStrategy
} // ModelInference