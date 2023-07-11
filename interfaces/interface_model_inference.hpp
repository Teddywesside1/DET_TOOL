#pragma once

#include "interface_model_framework.hpp"
#include "interface_dataloader.hpp"
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ModelInference{
    
class IModelInference{
protected:
    IModelInference(std::shared_ptr<ModelFramework::IModelFramework> & model_instance)
                : _model_instance(model_instance) {}

protected:
    std::shared_ptr<ModelFramework::IModelFramework> _model_instance {nullptr};
};


namespace ObjectDetection2D{

struct Object2D{
    int center_x;
    int center_y;
    int width;
    int height;
    int label;
    float confidence;
};

struct Anchor2D{
    int width;
    int height;
};

class IModelInferenceObjectDetection2D : public IModelInference{
public:
    virtual void 
    do_inference(const float conf_thresh,
                const std::shared_ptr<DataLoader::IDataLoader>& dataloader, 
                std::vector<std::vector<Object2D>> &output_objs) = 0;
protected:
    IModelInferenceObjectDetection2D(std::shared_ptr<ModelFramework::IModelFramework> & model_instance)
                                : IModelInference(model_instance) {}
};




} // ObjectDetection2D

} // ModelInference