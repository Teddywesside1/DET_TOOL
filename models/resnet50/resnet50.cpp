#include "CMakeConfig.h"

#include "resnet50.hpp"
#include "common.hpp"
#include <iostream>

using namespace std;

ResNet50::ResNet50(const int batch_size, IModelInstance* model_instance)
                :batch_size(batch_size), model_instance(model_instance){
    input_height = 224;
    input_width = 224;
    input_channel = 3;
}

void ResNet50::getInputHWC(int &input_height, int &input_width, int &input_channel){
    input_height = this->input_height;
    input_width = this->input_width;
    input_channel = this->input_channel;
}

void ResNet50::getBatchsize(int &batch_size){
    batch_size = this->batch_size;
}


// int ResNet50::inference(std::shared_ptr<float> data_ptr){
//     return model_instance->inference(data_ptr);
// }

// std::shared_ptr<float> ResNet50::getOneBatch(IDataLoader* dataloader_interface){
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



int ResNet50::inferenceRounds(IDataLoader* dataloader_interface, const int round){
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
