#include "hm800_model.hpp"
#include <iostream>
#include "common.hpp"


HM800ModelInstance::HM800ModelInstance(const string model_path, const string input_blob_name, const int batch_size,
                    const int input_height, const int input_width, const int input_channel, const int round, const int target_fps) : 
                input_blob_name(input_blob_name), module(tvm::hdpl::LoadModelPackage(model_path)), batch_size(batch_size), 
                input_height(input_height), input_width(input_width), input_channel(input_channel), round(round), delay_ms(1000.0/target_fps*batch_size*round) {
    runtime::NDArray image = runtime::NDArray::Empty({batch_size, input_height, input_width, input_channel}, DataType::Int(8), {kDLCPU, 0});
    module.SetInput(input_blob_name, image);
}

void HM800ModelInstance::inferenceRounds(vector<shared_ptr<float>>& roundsPtr){
    const char *env_platform_pointer = std::getenv("HDPL_PLATFORM");
    if (env_platform_pointer != nullptr && "ASIC" == string(env_platform_pointer)){
        module.RunRounds(round);
        hdplDeviceSynchronize();
    }else{
        Delay(delay_ms);
    }
    // module.RunRounds(round);
    // hdplDeviceSynchronize();
}
