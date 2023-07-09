#include "util_image.hpp"
#include <glog/logging.h>

ImageDrawHelper::ImageDrawHelper(const std::shared_ptr<cv::Mat> & image)
                                : _image(image){

}

void ImageDrawHelper::drawRect2D(const int center_x,
                const int center_y,
                const int width,
                const int height,
                const cv::Scalar color){
    cv::rectangle(*_image, 
                cv::Rect(center_x, center_y, width, height), 
                color, 
                DRAW_LINE_THICKNESS);
}

void ImageDrawHelper::drawRect2D(const Object2D& obj,
                const cv::Scalar color){
    drawRect2D(obj.center_x, obj.center_y, obj.width, obj.height, color);
}

