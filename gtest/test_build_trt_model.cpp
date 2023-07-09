#include "util_tensorrt.hpp"
#include <gtest/gtest.h>



TEST(build_model, tensorrt)
{
    using ModelFramework::TensorRT::create_engine_from_onnx;
    create_engine_from_onnx("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s.onnx");
}
