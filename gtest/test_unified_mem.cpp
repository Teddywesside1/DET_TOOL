#include "yolov5.hpp"
#include "tensorrt_framework.hpp"
#include "dataloader_objdet_2d.hpp"
#include <gtest/gtest.h>



TEST(unified_mem, alloc)
{
    const int N = 10;
    float* a;

    cudaMallocManaged((void **)&a, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
    }

    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    }

    cudaFree(a);
}