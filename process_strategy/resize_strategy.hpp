#include "interface_process_strategy.hpp"


namespace ModelInference{
namespace ProcessStrategy{

class ResizeStrategyPadding : public IResizeStrategy{
public:
    ResizeStrategyPadding() = delete;

    ResizeStrategyPadding(const int target_height, const int target_width);

    float operator()(const cv::Mat& src_image,
                    cv::Mat& dst_image) override;
};


// class ResizeStrategyPadding_CUDA : public IResizeStrategy{
// public:
//     ResizeStrategyPadding_CUDA() = delete;

//     ResizeStrategyPadding_CUDA(const int target_height, const int target_width);

//     float operator()(const cv::Mat& src_image,
//                     cv::Mat& dst_image) override;

// private:
//     std::shared_ptr<void> _buffer {nullptr};
// };


} // ProcessStrategy
} // ModelInference
