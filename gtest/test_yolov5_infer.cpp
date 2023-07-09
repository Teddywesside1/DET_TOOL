#include "yolov5.hpp"
#include "tensorrt_framework.hpp"
#include "dataloader_objdet_2d.hpp"
#include <gtest/gtest.h>
#include "util_image.hpp"

using namespace DataLoader;
using namespace ModelFramework::TensorRT;
using namespace ModelInference::ObjectDetection2D;

TEST(det_infer, yolov5)
{
    std::string test_image_path = "/data/binfeng/projects/server_multi-platform/images/bus.jpg";
    auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_trtexec_fp16.engine");
    Yolov5 yolo_model(model_instance, 640, 640, 3);
    const auto dataloader = std::make_shared<DataLoaderObjDet2D>();
    auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));
    for (int i = 0 ; i < 2000 ; ++ i ){
        dataloader->push(image);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<Object2D>> outputs;
        yolo_model.do_inference(dataloader, outputs);
        auto end = std::chrono::high_resolution_clock::now();    
        LOG(INFO) << "do_inference, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    // // auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));
    // ImageDrawHelper drawer(image);

    // for (const auto obj : outputs[0]){
    //     drawer.drawRect2D(obj);
    // }
    
    // cv::imwrite(test_image_path + "result.jpg", *image);
}