#include "CMakeConfig.h"
#include "yolov5.hpp"
#include <glog/logging.h>
#include <glog/log_severity.h>
#include "util_compute_function.hpp"
#include <chrono>

using namespace std;
namespace ModelInference{
namespace ObjectDetection2D{

Yolov5::Yolov5(std::shared_ptr<ModelFramework::IModelFramework> model_instance,
            const int input_height,
            const int input_width,
            const int input_channel,
            const int cls_number) 
            : IModelInferenceObjectDetection2D(model_instance)
            , _input_height(input_height)
            , _input_width(input_width)
            , _input_channel(input_channel)
            , _cls_number(cls_number)
            , _resize_strategy(input_height, input_width)
            , _flatten_strategy(){

    _nms_thresh = 0.45;
}

void Yolov5::do_inference(const float conf_thresh,
                        const std::shared_ptr<DataLoader::IDataLoader>& dataloader, 
                        std::vector<std::vector<Object2D>> &output_objs){
    // 1. get the infer data buffer 
    std::vector<void*> &buffers = _model_instance->get_buffer();
    CHECK(buffers.size() == 4);

    // 2. preprocess images and store scale_info
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<DataLoader::BatchInfo> image_scale_info;
    const int batch_size = pre_process(dataloader, image_scale_info, buffers[0]);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "pre_process, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 3. infer by infer_framework
    start = std::chrono::high_resolution_clock::now();
    _model_instance->framework_forward(batch_size);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "framework_forward, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 4. postprocess
    start = std::chrono::high_resolution_clock::now();
    std::vector<float*> output_blobs(3);
    for (int i = 1 ; i < buffers.size() ; ++ i){
        output_blobs[i-1] = static_cast<float*>(buffers[i]);
    }
    output_objs.clear();
    output_objs.resize(batch_size);
    post_process(batch_size, conf_thresh, output_blobs, output_objs);
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "post_process, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 5. scale to origin image size
    scale_objs_to_origin_image_size(output_objs, image_scale_info);
}



int Yolov5::pre_process(const std::shared_ptr<DataLoader::IDataLoader>& dataloader,
                        std::vector<DataLoader::BatchInfo> &image_scale_info,
                        void * input_blob){
    // const int single_channel_pixel_size = _input_height * _input_width;
    // const int single_image_float_element_size = single_channel_pixel_size * _input_channel;
    std::vector<float *> input_blobs(1);
    input_blobs[0] = static_cast<float*>(input_blob);

    // auto resize_strategy = 
    //         [&](const cv::Mat& image){
    //             const int ori_height = image.rows;
    //             const int ori_width = image.cols;
    //             int fix_height, fix_width;
    //             float scale;
    //             if (ori_height > ori_width){
    //                 fix_height = _input_height;
    //                 scale = _input_height / static_cast<float>(ori_height);
    //                 fix_width = static_cast<int>(ori_width * scale);
    //             }else{
    //                 fix_width = _input_width;
    //                 scale = _input_width / static_cast<float>(ori_width);
    //                 fix_height = static_cast<int>(ori_height * scale);
    //             }
    //             image_scale_info.push_back(ImageScaleInfo{ori_height, ori_width, scale});

    //             cv::Mat resized_image;
    //             cv::resize(image, resized_image, {fix_width, fix_height});
    //             auto out = std::make_shared<cv::Mat>(_input_height, _input_width, CV_8UC3, cv::Scalar{0, 0, 0});
    //             resized_image.copyTo((*out)(cv::Rect(0, 0, fix_width, fix_height)));
    //             return out;
    //         };

    // auto flatten_strategy = 
    //         [&](float* data_ptr, const cv::Mat& image){
    //             int idx = 0;
    //             for (int r = 0 ; r < _input_height ; ++ r){
    //                 uchar* pixel_ptr = image.data + r * image.step;
    //                 for (int c = 0 ; c < _input_width ; ++ c){
    //                     data_ptr[idx] = pixel_ptr[2] / 255.0;
    //                     data_ptr[idx + single_channel_pixel_size] = pixel_ptr[1] / 255.0;
    //                     data_ptr[idx + 2 * single_channel_pixel_size] = pixel_ptr[0] / 255.0;
    //                     pixel_ptr += 3;
    //                     ++ idx;
    //                     // if (data_ptr > input_blobs[0]) 
    //                     //     LOG(INFO) << "buffer data : " << data_ptr[idx] << ",  pixel value : " << pixel_ptr[2] / 255.0;
    //                 }
    //             }
    //             return single_image_float_element_size;
    //         };


    return dataloader->get_buffer_as_one_batch(_resize_strategy,
                                _flatten_strategy,
                                input_blobs,
                                image_scale_info);    

}


void Yolov5::scale_objs_to_origin_image_size(
                        std::vector<std::vector<Object2D>> &output_objs,
                        std::vector<DataLoader::BatchInfo> &image_scale_info){
    CHECK(output_objs.size() == image_scale_info.size());

    const int batch_size = image_scale_info.size();
    for (int i = 0 ; i < batch_size ; ++ i){
        const DataLoader::BatchInfo & info = image_scale_info[i];
        // LOG(INFO) << "image_scale_info, batch_idx : " << i << " scale : " << info.scale;
        auto & objects = output_objs[i];
        for (auto & obj : objects){
            obj.center_x = static_cast<int>(obj.center_x / info.scale);
            obj.center_y = static_cast<int>(obj.center_y / info.scale);
            obj.height = static_cast<int>(obj.height / info.scale);
            obj.width = static_cast<int>(obj.width / info.scale);
        }
    }
}


void Yolov5::post_process(const int batch_size,
                const float conf_thresh,
                std::vector<float*> &output_blobs,
                std::vector<std::vector<Object2D>> &output_objs){

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Object2D>> candidate_objs(batch_size);
    const int LEVEL_TOTAL = output_blobs.size();
    for (int level = 0 ; level < LEVEL_TOTAL ; ++ level){
        generate_anchor_from_blob(batch_size,
                                level,
                                conf_thresh,
                                output_blobs[level],
                                candidate_objs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "post_process -> generate candidates, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> picked_idxes(batch_size);
    for (int batch_idx = 0 ; batch_idx < batch_size ; ++ batch_idx){
        nms(candidate_objs[batch_idx], picked_idxes[batch_idx]);
    }
    end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "post_process -> nms, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    /*
        TODO:
            map the bbox to origin image size
    */
    for (int batch_idx = 0 ; batch_idx < batch_size ; ++ batch_idx){
        output_objs[batch_idx].resize(picked_idxes[batch_idx].size());
        int i = 0;
        for (const int picked_idx : picked_idxes[batch_idx]){
            output_objs[batch_idx][i++] = candidate_objs[batch_idx][picked_idx];
        }
    }
}



void Yolov5::generate_anchor_from_blob( const int batch_size,
                                const int level,
                                const float conf_thresh,
                                float* output_blob,
                                std::vector<std::vector<Object2D>> &candidate_objs){
    CHECK(level >= 0 && level < anchors.size());
    const std::vector<Anchor2D> &level_anchor = anchors[level];
    const int scale_factor = scale_factors[level];

    const int level_height = _input_height / scale_factor;
    const int level_width = _input_width / scale_factor;
    const int level_anchor_number = level_anchor.size();
    const int features_per_anchor = 5 + _cls_number;
    const int total_float_element_number_per_batch = level_height
                                        * level_width
                                        * level_anchor_number 
                                        * features_per_anchor;
    const int float_element_number_per_anchor = level_height
                                            * level_width
                                            * features_per_anchor;
    /*
        data in pointer `output_blob` is a flatten of 
            tensor [batch_size, anchor_number, height, width, features_per_anchor]
    */
    const int level_total_pixel = level_height * level_width;

    for (int batch_idx = 0; batch_idx < batch_size ; ++ batch_idx){
        const int offset_in_batch = total_float_element_number_per_batch * batch_idx;
        for (int anchor_idx = 0 ; anchor_idx < level_anchor_number ; ++ anchor_idx){
            const int anchor_width = level_anchor[anchor_idx].width;
            const int anchor_height = level_anchor[anchor_idx].height;
            const int offset_in_anchor = offset_in_batch + float_element_number_per_anchor * anchor_idx;
            for (int f_r = 0 ; f_r < level_height ; ++ f_r){
                for (int f_c = 0 ; f_c < level_width ; ++ f_c){
                    // 1. get the feature for target anchor at [anchor_idx, f_r, f_c]
                    const int offset_in_pixel = offset_in_anchor + (f_r * level_width + f_c) * features_per_anchor;
                    const float * tmp = &output_blob[offset_in_pixel];
                    // memcpy(tmp.data(), &output_blob[offset_in_pixel], sizeof(float) * features_per_anchor);

                    // 2. get most likely class
                    int class_idx = 0;
                    float class_score = tmp[5];
                    for (int i = 1 ; i < _cls_number ; ++ i){
                        if (tmp[5 + i] > class_score){
                            class_score = tmp[5 + i];
                            class_idx = i;
                        }
                    }
                    // 3. get confidence
                    const float confidence = sigmoid(class_score) * sigmoid(tmp[4]);
                    // if (batch_idx == 1) LOG(INFO) << "confidence : " << confidence;
                    // 4. compute bbox
                    if (confidence > conf_thresh){
                        const float dx = sigmoid(tmp[0]);
                        const float dy = sigmoid(tmp[1]);
                        const float dw = sigmoid(tmp[2]);
                        const float dh = sigmoid(tmp[3]);

                        const float bbox_width = pow(dw * 2.f, 2) * anchor_width;
                        const float bbox_height = pow(dh * 2.f, 2) * anchor_height;

                        const float center_x = (f_c + dx * 2.f - 0.5f) * scale_factor - bbox_width * 0.5;
                        const float center_y = (f_r + dy * 2.f - 0.5f) * scale_factor - bbox_height * 0.5;

                        candidate_objs[batch_idx].push_back({
                            static_cast<int>(center_x),
                            static_cast<int>(center_y),
                            static_cast<int>(bbox_width),
                            static_cast<int>(bbox_height),
                            class_idx,
                            confidence
                        });
                    }
                }   // f_c
            }   // f_r
        }   // anchor_idx
    }   // batch_idx

}

void Yolov5::nms(const std::vector<Object2D> &candidate_objs,
                std::vector<int>& picked_idxes){
    const int candidate_objs_total = candidate_objs.size();
    // LOG(INFO) << "nms : candidate_objs_total " << candidate_objs_total;
    for (int i = 0 ; i < candidate_objs_total ; ++ i){
        const Object2D & obj_a = candidate_objs[i];
        
        bool keep = true;
        for (const int picked_idx : picked_idxes){
            const Object2D & obj_b = candidate_objs[picked_idx];
            const float inter_area_size = rect_inter_area_size(obj_a, obj_b);
            const float union_area_size = obj_a.width * obj_a.height
                                        + obj_b.width * obj_b.height
                                        - inter_area_size;
            
            if (inter_area_size / union_area_size > _nms_thresh){
                keep = false;
                break;
            }
        }

        if (keep){
            picked_idxes.push_back(i);
        }
    }
}



} // ObjectDetection2D
} // ModelInference