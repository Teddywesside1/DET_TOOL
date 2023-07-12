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

} // ProcessStrategy
} // ModelInference
