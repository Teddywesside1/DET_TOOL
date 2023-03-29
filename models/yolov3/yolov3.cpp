#include "CMakeConfig.h"
#include "yolov3.hpp"
#include "common.hpp"
#include <iostream>

using namespace std;

Yolov3::Yolov3(const int batch_size, IModelInstance* model_instance)
                :batch_size(batch_size), model_instance(model_instance){
    input_height = 416;
    input_width = 416;
    input_channel = 3;
}

void Yolov3::getInputHWC(int &input_height, int &input_width, int &input_channel){
    input_height = this->input_height;
    input_width = this->input_width;
    input_channel = this->input_channel;
}

void Yolov3::getBatchsize(int &batch_size){
    batch_size = this->batch_size;
}

// int Yolov3::inference(std::shared_ptr<float> data_ptr){
//     return model_instance->inference(data_ptr);
// }

// std::shared_ptr<float> Yolov3::getOneBatch(IDataLoader* dataloader_interface){
//     int single_channel_pixel_size = input_height * input_width;
//     int single_input_pixel_size =  single_channel_pixel_size * input_channel;

// #if (WITHOUT_IMAGE_READING)
//     return dataloader_interface->getOneBatch(batch_size, single_input_pixel_size);
// #else
//     return dataloader_interface->getOneBatch(batch_size, single_input_pixel_size, [&](std::shared_ptr<float> data_ptr, 
//                                     cv::Mat& image, const int image_count){
//         cv::resize(image,image,{input_width,input_height});
//         int idx = 0, offset = image_count * single_input_pixel_size;
//         for (int r = 0 ; r < input_height ; ++ r){
//             uchar* pixel_ptr = image.data + r * image.step;
//             for (int c = 0 ; c < input_width ; ++ c){
//                 data_ptr.get()[offset + idx] = pixel_ptr[2] / 255.0;
//                 data_ptr.get()[offset + idx + single_channel_pixel_size] = pixel_ptr[1] / 255.0;
//                 data_ptr.get()[offset + idx + 2 * single_channel_pixel_size] = pixel_ptr[0] / 255.0;
//                 pixel_ptr += 3;
//                 ++ idx;
//             }
//         }
//     });
// #endif
// }



int Yolov3::inferenceRounds(IDataLoader* dataloader_interface, const int round){
    const int single_channel_pixel_size = input_height * input_width;
    const int single_input_pixel_size =  single_channel_pixel_size * input_channel;

#if (WITHOUT_IMAGE_READING)
    auto roundsPtr = dataloader_interface->getOneRound(round, batch_size, single_input_pixel_size);
#else
    auto roundsPtr = dataloader_interface->getOneRound(round, batch_size, single_input_pixel_size, [&](std::shared_ptr<float> data_ptr, 
                                    cv::Mat& image, const int image_count){
        cv::resize(image,image,{input_width,input_height});
        int idx = 0, offset = image_count * single_input_pixel_size;
        for (int r = 0 ; r < input_height ; ++ r){
            uchar* pixel_ptr = image.data + r * image.step;
            for (int c = 0 ; c < input_width ; ++ c){
                data_ptr.get()[offset + idx] = pixel_ptr[2] / 255.0;
                data_ptr.get()[offset + idx + single_channel_pixel_size] = pixel_ptr[1] / 255.0;
                data_ptr.get()[offset + idx + 2 * single_channel_pixel_size] = pixel_ptr[0] / 255.0;
                pixel_ptr += 3;
                ++ idx;
            }
        }
    });
#endif

    if (roundsPtr.size() == 0) return -1;

    model_instance->inferenceRounds(roundsPtr);
    return round * batch_size;
}
