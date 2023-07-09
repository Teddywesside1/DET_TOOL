#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "interface_model_inference.hpp"

using ModelInference::ObjectDetection2D::Object2D;


#define DRAW_LINE_THICKNESS 2

class ImageDrawHelper {
public:
    ImageDrawHelper() = delete;
    ImageDrawHelper(const std::shared_ptr<cv::Mat> & image);

    void drawRect2D(const int center_x,
                    const int center_y,
                    const int width,
                    const int height,
                    const cv::Scalar color = {255, 0, 0});
    
    void drawRect2D(const Object2D& obj,
                    const cv::Scalar color = {255, 0, 0});


private:
    std::shared_ptr<cv::Mat> _image;
};
