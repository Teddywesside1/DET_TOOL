#include "util_compute_function.hpp"
#include <cmath>
#include <glog/logging.h>



float
rect_inter_area_size(const Object2D & obj_a, const Object2D & obj_b){
    cv::Rect_<float> inter = cv::Rect_<float>(obj_a.center_x, 
                                        obj_a.center_y, 
                                        obj_a.width, 
                                        obj_a.height)
                            & cv::Rect_<float>(obj_b.center_x, 
                                        obj_b.center_y, 
                                        obj_b.width, 
                                        obj_b.height);
    return inter.area();
}