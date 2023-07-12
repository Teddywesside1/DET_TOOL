#include "yolov5.hpp"
#include "tensorrt_framework.hpp"
#include "dataloader_objdet_2d.hpp"
#include <gtest/gtest.h>
#include "util_image.hpp"

using namespace DataLoader;
using namespace ModelFramework::TensorRT;
using namespace ModelInference::ObjectDetection2D;

const float CONF_THRESH = 0.3;
const int CLASSIFICATION_NUMBER = 80;

TEST(det_infer_single_batch_one_run, yolov5)
{
    std::string test_image_path = "/data/binfeng/projects/server_multi-platform/images/bus.jpg";
    auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_trtexec_fp16.engine");
    Yolov5 yolo_model(model_instance, 640, 640, 3, CLASSIFICATION_NUMBER);
    const auto dataloader = std::make_shared<DataLoaderObjDet2D>();
    auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));
    dataloader->push(image);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Object2D>> outputs;
    yolo_model.do_inference(CONF_THRESH, dataloader, outputs);
    auto end = std::chrono::high_resolution_clock::now();    
    LOG(INFO) << "do_inference, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    ImageDrawHelper drawer(image);

    for (const auto obj : outputs[0]){
        drawer.drawRect2D(obj);
    }
    
    cv::imwrite(test_image_path + "_test_result.jpg", *image);
}

TEST(det_infer_single_batch_many_run, yolov5)
{
    std::string test_image_path = "/data/binfeng/projects/server_multi-platform/images/bus.jpg";
    auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_trtexec_fp16.engine");
    Yolov5 yolo_model(model_instance, 640, 640, 3, CLASSIFICATION_NUMBER);
    const auto dataloader = std::make_shared<DataLoaderObjDet2D>();
    auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));
    for (int i = 0 ; i < 100 ; ++ i ){
        dataloader->push(image);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<Object2D>> outputs;
        yolo_model.do_inference(CONF_THRESH, dataloader, outputs);
        auto end = std::chrono::high_resolution_clock::now();    
        LOG(INFO) << "do_inference, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
}




TEST(det_infer_multi_batch_one_run, yolov5)
{
    const int BATCH_SIZE = 4;

    const std::string test_image_path = "/data/binfeng/projects/server_multi-platform/images/bus.jpg";
    auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_dynamic_batch_fp16.engine");
    Yolov5 yolo_model(model_instance, 640, 640, 3, CLASSIFICATION_NUMBER);
    const auto dataloader = std::make_shared<DataLoaderObjDet2D>();
    auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));

    for (int i = 0 ; i < BATCH_SIZE ; ++ i)
        dataloader->push(image);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Object2D>> outputs;
    yolo_model.do_inference(CONF_THRESH, dataloader, outputs);
    auto end = std::chrono::high_resolution_clock::now();    
    LOG(INFO) << "do_inference, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    for (int i = 0 ; i < BATCH_SIZE ; ++ i){
        auto image_to_draw_ptr = std::make_shared<cv::Mat>(cv::imread(test_image_path));
        ImageDrawHelper drawer(image_to_draw_ptr);

        for (const auto obj : outputs[i]){
            drawer.drawRect2D(obj);
        }
        
        cv::imwrite(test_image_path +"_" + to_string(i) + "_test_result.jpg", *image_to_draw_ptr);        
    }

}





static void multi_batch_infer(const std::string test_image_path,
                            Yolov5 &yolo_model,
                            const int BATCH_SIZE){
    const auto dataloader = std::make_shared<DataLoaderObjDet2D>();
    auto image = std::make_shared<cv::Mat>(cv::imread(test_image_path));

    for (int i = 0 ; i < BATCH_SIZE ; ++ i)
        dataloader->push(image);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Object2D>> outputs;
    yolo_model.do_inference(CONF_THRESH, dataloader, outputs);
    auto end = std::chrono::high_resolution_clock::now();    
    LOG(INFO) << "do_inference, cost : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    for (int i = 0 ; i < BATCH_SIZE ; ++ i){
        auto image_to_draw_ptr = std::make_shared<cv::Mat>(cv::imread(test_image_path));
        ImageDrawHelper drawer(image_to_draw_ptr);

        for (const auto obj : outputs[i]){
            drawer.drawRect2D(obj);
        }
        
        cv::imwrite(test_image_path +"_" + to_string(i) + "_test_result.jpg", *image_to_draw_ptr);        
    }
}


TEST(det_infer_multi_batch_changing, yolov5)
{
    const std::string test_image_path_1 = "/data/binfeng/projects/server_multi-platform/images/bus.jpg";
    const std::string test_image_path_2 = "/data/binfeng/projects/server_multi-platform/images/car.jpg";
    auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_dynamic_batch_fp16.engine");
    Yolov5 yolo_model(model_instance, 640, 640, 3, CLASSIFICATION_NUMBER);

    {
        multi_batch_infer(test_image_path_1, yolo_model, 2);
    }
    {
        multi_batch_infer(test_image_path_2, yolo_model, 4);
    }
}
