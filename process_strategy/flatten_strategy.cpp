#include "flatten_strategy.hpp"

namespace ModelInference{
namespace ProcessStrategy{


int 
FlattenStrategyHWC2CHW::operator()(float* data_ptr, const cv::Mat& image){
    const int height = image.rows, width = image.cols, channel = image.channels(); 
    const int single_channel_pixel_size = height * width;
    const int single_image_float_element_size = single_channel_pixel_size * channel;
    int idx = 0;
    for (int r = 0 ; r < height ; ++ r){
        uchar* pixel_ptr = image.data + r * image.step;
        for (int c = 0 ; c < width ; ++ c){
            data_ptr[idx] = pixel_ptr[2] / 255.0;
            data_ptr[idx + single_channel_pixel_size] = pixel_ptr[1] / 255.0;
            data_ptr[idx + 2 * single_channel_pixel_size] = pixel_ptr[0] / 255.0;
            pixel_ptr += 3;
            ++ idx;
        }
    }
    return single_image_float_element_size;
}



} // ProcessStrategy
}// ModelInference