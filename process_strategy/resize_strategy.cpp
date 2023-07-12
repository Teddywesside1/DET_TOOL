#include "resize_strategy.hpp"


namespace ModelInference{
namespace ProcessStrategy{


ResizeStrategyPadding::ResizeStrategyPadding(const int target_height,
                                            const int target_width)
                                            : IResizeStrategy(target_height, target_width){

}

float
ResizeStrategyPadding::operator()(const cv::Mat& src_image,
                                cv::Mat& dst_image){
    const int ori_height = src_image.rows;
    const int ori_width = src_image.cols;
    int fix_height, fix_width;
    float scale;
    if (ori_height > ori_width){
        fix_height = _target_height;
        scale = _target_height / static_cast<float>(ori_height);
        fix_width = static_cast<int>(ori_width * scale);
    }else{
        fix_width = _target_width;
        scale = _target_width / static_cast<float>(ori_width);
        fix_height = static_cast<int>(ori_height * scale);
    }

    cv::Mat resized_image;
    cv::resize(src_image, resized_image, {fix_width, fix_height});
    dst_image = cv::Mat(_target_height, _target_width, CV_8UC3, cv::Scalar{0, 0, 0});
    resized_image.copyTo(dst_image(cv::Rect(0, 0, fix_width, fix_height)));
    return scale;
}


} // ProcessStrategy
} // ModelInference
