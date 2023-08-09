#include "ncnn_framework.hpp"
#include "dataloader_objdet_2d.hpp"
#include <gtest/gtest.h>
#include "util_image.hpp"

using namespace ModelFramework::NCNN;

TEST(framework_ncnn, build_buffer)
{
    NCNNModelFramework ncnn_framework("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_dynamic_batch_sim.param",
                                    "/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_dynamic_batch_sim.bin");

    // NCNNModelFramework ncnn_framework("/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_sim.param",
    //                                 "/data/binfeng/projects/server_multi-platform/pretrained/yolov5s_sim.bin");

}
