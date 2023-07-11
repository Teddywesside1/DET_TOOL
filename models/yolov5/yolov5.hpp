#pragma once

// #include "model_instance.hpp"
#include "interface_model_inference.hpp"
namespace ModelInference{
namespace ObjectDetection2D{

class Yolov5 : public IModelInferenceObjectDetection2D{    
    struct ImageScaleInfo{
        int origin_height;
        int origin_width;
        float scale;
    };

public:
    Yolov5(std::shared_ptr<ModelFramework::IModelFramework> model_instance,
        const int input_height,
        const int input_width,
        const int input_channel);
public:
    void do_inference(const float conf_thresh,
                    const std::shared_ptr<DataLoader::IDataLoader>& dataloader, 
                    std::vector<std::vector<Object2D>> &output_objs) override;

private:
    int pre_process(const std::shared_ptr<DataLoader::IDataLoader>& dataloader,
                    std::vector<ImageScaleInfo> &image_scale_info,
                    void * input_blob);


    void post_process(const int batch_size,
                    const float conf_thresh,
                    std::vector<float*> &output_blobs,
                    std::vector<std::vector<Object2D>> &output_objs);

    void scale_objs_to_origin_image_size(
                        std::vector<std::vector<Object2D>> &output_objs,
                        std::vector<ImageScaleInfo> &image_scale_info);

    void generate_anchor_from_blob( const int batch_size,
                                    const int level,
                                    const float conf_thresh,
                                    float* output_blob,
                                    std::vector<std::vector<Object2D>> &output_objs);

    void nms(const std::vector<Object2D> &candidate_objs,
            std::vector<int>& picked_idx);

private:
    int _input_height {0};
    int _input_width {0};
    int _input_channel {0};
    int _cls_number {0};

    float _nms_thresh {0};

    std::vector<std::shared_ptr<float>> _input_blobs{};

    std::vector<std::vector<Anchor2D>> anchors {
        {{10,13}, {16,30}, {33,23}},
        {{30,61}, {62,45}, {59,119}},
        {{116,90}, {156,198}, {373,326}}
    };

    std::vector<int> scale_factors {
        8, 16, 32
    };
};

// class Yolov3Factory : public ModelRuntimeFactoryBase{
// public:
//     Yolov3Factory(const int batch_size, IModelInstance* model_instance) :
//                 batch_size(batch_size), model_instance(model_instance){

//     }
    
//     ModelRuntime* createRuntimeModel(){
//         return new Yolov3(batch_size,model_instance);
//     }
// private:
//     int batch_size;
//     IModelInstance* model_instance;
// };

} // ObjectDetection2D

} // ModelInference