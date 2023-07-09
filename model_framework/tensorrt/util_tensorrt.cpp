#include "util_tensorrt.hpp"
#include <fstream>
// #include "tensorrt_engine_config.hpp"

namespace ModelFramework{

namespace TensorRT{

using namespace nvinfer1;

static std::shared_ptr<IHostMemory> parse_onnx_file(const std::string& model_path, 
                            TensorrtLogger &logger)
{
    std::unique_ptr<IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(1)};
    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX

    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {   
        throw TRTCreateEngineException("parse onnx file failed !");
    }
    // IOptimizationProfile* profile = builder->createOptimizationProfile();
    // model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kMIN, Dims4(1,3,input_height,input_width));
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kOPT, Dims4(32,3,input_height,input_width));
    // profile->setDimensions(input_blob_name.c_str(),OptProfileSelector::kMAX, Dims4(32,3,input_height,input_width));
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    // model_config->nvConfigSetting(config.get(), batch_size, input_height, input_width, input_channel);
    // config->addOptimizationProfile(profile);
    // generate TensorRT engine optimized for the target platform
    // builder->setMaxBatchSize(32);
    std::shared_ptr<nvinfer1::IHostMemory> engineOutput;
    engineOutput.reset(builder->buildSerializedNetwork(*network, *config));
    // saveEngine(model_path + ".engine", engineOutput);
    return engineOutput;
}

static void saveEngine(const std::string& outputFilePath, std::shared_ptr<nvinfer1::IHostMemory> engine_serialized)
{
    /*  Write to disk   */    
    std::ofstream outputFile;
    outputFile.open(outputFilePath, std::ios::out | std::ios::binary);
    outputFile.write((char*)engine_serialized->data(), engine_serialized->size());
    if(!outputFile.good())
    {
        throw TRTCreateEngineException("save engine failed !");
    }
    outputFile.close();
}


void 
create_engine_from_onnx(const std::string &onnx_file_path){
    TensorrtLogger logger;
    saveEngine(onnx_file_path + ".engine", 
                parse_onnx_file(onnx_file_path, logger));
}




} // TensorRT

} // ModelFramework
