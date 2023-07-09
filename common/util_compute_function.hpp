#pragma once

#include "interface_model_inference.hpp"

using ModelInference::ObjectDetection2D::Object2D;

inline float
sigmoid(const float x){
    return 1.f / (1.f + exp(-x));
}


float
rect_inter_area_size(const Object2D & obj_a, const Object2D & obj_b);