#include "TRT_model.hpp"


#include "common.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>

using namespace std;
using namespace nvinfer1;

TRTModelInstance::TRTModelInstance(const string model_path, const string input_blob_name, const int batch_size, 
            const int input_height, const int input_width, const int input_channel, shared_ptr<ITRTModelConfig> model_config)
        : input_blob_name(input_blob_name), batch_size(batch_size), input_height(input_height), 
        input_width(input_width), input_channel(input_channel), model_config(model_config) { 
    string model_file_type = getSuffix(model_path);
    if ("onnx" == model_file_type){
        parseOnnxModel(model_path);
    }else if ("engine" == model_file_type){
        loadEngine(model_path);
    }
    cout << "loading engine finished!" << endl;
    prepareContext();
    cudaStreamCreate(&stream);
    if (model_config->isDynamicBatchSize()){
        context->setBindingDimensions(0,Dims4(batch_size,input_channel,input_height,input_width));
    }
}

void TRTModelInstance::parseOnnxModel(const string& model_path)
{
    unique_ptr<IBuilder> builder{nvinfer1::createInferBuilder(tlogger)};
    unique_ptr<INetworkDefinition> network{builder->createNetworkV2(1)};
    unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, tlogger)};
    unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX

    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {   
        throw runtime_error(" -- parsing from onnx_file failed ! ");
    }
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kMIN, Dims4(1,3,input_height,input_width));
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kOPT, Dims4(32,3,input_height,input_width));
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kMAX, Dims4(32,3,input_height,input_width));
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    model_config->nvConfigSetting(config.get(), batch_size, input_height, input_width, input_channel);
    config->addOptimizationProfile(profile);
    // generate TensorRT engine optimized for the target platform
    // builder->setMaxBatchSize(32);
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    unique_ptr<nvinfer1::IHostMemory> engineOutput;
    engineOutput.reset(builder->buildSerializedNetwork(*network, *config));
    saveEngine(model_path + ".engine", engineOutput);

}

void TRTModelInstance::saveEngine(const string& outputFilePath, unique_ptr<nvinfer1::IHostMemory>& engineOutput)
{
    /*  Write to disk   */    
    std::ofstream outputFile;
    outputFile.open(outputFilePath, std::ios::out | std::ios::binary);
    outputFile.write((char*)engineOutput->data(), engineOutput->size());
    if(!outputFile.good())
    {
        throw runtime_error(" -- engine saving failed ! ");
    }
    outputFile.close();
}


void TRTModelInstance::loadEngine(const string& engine_file_path) {
    ifstream file(engine_file_path, ios::binary);
    if(!file.good())
    {
        throw runtime_error(" -- engine loading failed ! ");
    }

    std::vector<char> data;

    file.seekg(0, file.end);
    const auto size = file.tellg();
    file.seekg(0, file.beg);

    data.resize(size);
    file.read(data.data(), size);
    
    file.close();

    unique_ptr<IRuntime> runtime{nvinfer1::createInferRuntime(tlogger)};

    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
}



void TRTModelInstance::prepareContext(){
    context.reset(engine->createExecutionContext());

    int bd_number = engine->getNbBindings();
    buffers.resize(bd_number);
    for (int i = 0 ; i < bd_number ; i++)
	{
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
		int dim_byte_size = getSizeByDim(dims) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i],dim_byte_size);
	}   
}


int TRTModelInstance::getSizeByDim(const nvinfer1::Dims& dims){
    int ret = 1;
    for (int i = 1; i < dims.nbDims; ++ i)
	{
		ret *= dims.d[i];
	}
    return ret;
}



void TRTModelInstance::inferenceRounds(vector<shared_ptr<float>>& roundsPtr){
    int input_byte_size = input_height * input_width * input_channel * batch_size * sizeof(float);
    for (auto& data_ptr : roundsPtr){
        if (!model_config->noDataTransfers())
            cudaMemcpyAsync(buffers[0],data_ptr.get(),input_byte_size,cudaMemcpyHostToDevice,stream);
        // context->enqueue(batch_size,buffers.data(),stream,NULL);
        context->enqueueV2(buffers.data(),stream,NULL);
        cudaStreamSynchronize(stream);    
    }
}