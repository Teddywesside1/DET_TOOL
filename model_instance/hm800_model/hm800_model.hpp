#pragma once
#include "model_instance.hpp"
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include "hdpl/hdpl_runtime.h"
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <memory>

using namespace std;
using namespace tvm;
using namespace tvm::relay;



class HM800ModelInstance : public IModelInstance {
public:
    HM800ModelInstance(const string model_path, const string input_blob_name, const int batch_size,
                const int input_height, const int input_width, const int input_channel, const int round, const int target_fps);
public:
    void inferenceRounds(vector<shared_ptr<float>>& data_ptr) override;

private:
    tvm::hdpl::Module module;
    const int batch_size, round;
    int input_height, input_width, input_channel;
    const string input_blob_name;

    const int delay_ms;
};