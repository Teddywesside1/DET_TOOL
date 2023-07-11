#include "util_tensorrt.hpp"
#include "tensorrt_framework.hpp"
#include <gtest/gtest.h>

using namespace ModelFramework::TensorRT;

// TEST(build_model, tensorrt)
// {
//     using ModelFramework::TensorRT::create_engine_from_onnx;
//     create_engine_from_onnx("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s.onnx");
// }



TEST(deconstructor, tensorrt)
{
    {    
        auto model_instance = std::make_shared<TRTModelFramework>("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_trtexec_fp16.engine");
    }    
    
}