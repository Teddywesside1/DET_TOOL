#include "tensorrt_framework.hpp"


#include "common.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include "util_path.hpp"
#include <glog/logging.h>
#include <glog/log_severity.h>


using namespace nvinfer1;

namespace ModelFramework{

namespace TensorRT{

TRTModelFramework::TRTModelFramework(const std::string model_path) { 
    std::string model_file_type = get_suffix(model_path);
    if ("engine" != model_file_type){
        throw TRTRuntimeException("got model path : " + model_path + ", which should end with `.engine` suffix");
    }
    loadEngine(model_path);
    prepareContext();
    // cudaStreamCreate(&stream);

    _engine_built_with_explicit_batch = engine->hasImplicitBatchDimension();
}


void TRTModelFramework::loadEngine(const string& engine_file_path) {
    ifstream file(engine_file_path, ios::binary);
    if(!file.good())
    {
        throw TRTRuntimeException("engine file : " + engine_file_path + " not good !");
    }

    std::vector<char> data;

    file.seekg(0, file.end);
    const auto size = file.tellg();
    file.seekg(0, file.beg);

    data.resize(size);
    file.read(data.data(), size);
    
    file.close();

    unique_ptr<IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
}



void TRTModelFramework::prepareContext(){
    context.reset(engine->createExecutionContext());

    int bd_number = engine->getNbBindings();
    buffers.resize(bd_number);
    for (int i = 0 ; i < bd_number ; i++)
	{
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
        if (i > 0)
            LOG(INFO) << "blob name : " << engine->getBindingName(i) 
                        << " dims : " 
                        << dims.d[0] << " "
                        << dims.d[1] << " "
                        << dims.d[2] << " "
                        << dims.d[3] << " "
                        << dims.d[4];
		int dim_byte_size = getSizeByDim(dims) * MAX_BATCH_SIZE * sizeof(float);
        // cudaMalloc(&buffers[i], dim_byte_size);
        auto ret = cudaMallocManaged(&buffers[i], dim_byte_size);
        CHECK(ret == cudaSuccess);
        cudaDeviceSynchronize();
	}   
}


int TRTModelFramework::getSizeByDim(const nvinfer1::Dims& dims){
    int ret = 1;
    for (int i = 1; i < dims.nbDims; ++ i)
	{
		ret *= dims.d[i];
	}
    return ret;
}


std::vector<void*>& TRTModelFramework::get_buffer(){
    return buffers;
}


void TRTModelFramework::framework_forward(const int batch_size){
    if (!_engine_built_with_explicit_batch){    
        int bd_number = engine->getNbBindings();
        for (int i = 0 ; i < bd_number ; i++)
        {
            if (!engine->bindingIsInput(i)) continue;
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            if (dims.d[0] == batch_size) continue;
            dims.d[0] = batch_size;
            context->setBindingDimensions(i, dims);
        }  
    }
    CHECK(context->executeV2(buffers.data()));
    // CHECK(context->execute(batch_size, buffers.data()));
}

// void TRTModelFramework::inferenceRounds(vector<shared_ptr<float>>& roundsPtr){
//     int input_byte_size = input_height * input_width * input_channel * batch_size * sizeof(float);
//     for (auto& data_ptr : roundsPtr){
//         if (!model_config->noDataTransfers())
//             cudaMemcpyAsync(buffers[0],data_ptr.get(),input_byte_size,cudaMemcpyHostToDevice,stream);
//         // context->enqueue(batch_size,buffers.data(),stream,NULL);
//         context->enqueueV2(buffers.data(),stream,NULL);
//         cudaStreamSynchronize(stream);    
//     }
// }

} // TensorRT

} // ModelFramework