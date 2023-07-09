#pragma once

#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace ModelFramework{

namespace TensorRT{

using namespace nvinfer1;

class ITRTFrameworkCreateEngineFromOnnxConfig {
public:
    virtual void nvProfileSetting(IOptimizationProfile* profile) = 0;
    virtual void nvConfigSetting(IBuilderConfig* config) = 0;
    virtual bool isDynamicBatchSize() = 0;
    virtual ~ITRTFrameworkCreateEngineFromOnnxConfig(){}

};

using ITRTEngineConfig = ITRTFrameworkCreateEngineFromOnnxConfig;

class TRTEngineConfig_Default : public ITRTEngineConfig{
public:
    TRTEngineConfig_Default(std::shared_ptr<ITRTEngineConfig> engine_config) 
    : _engine_config(engine_config) {}

    void nvConfigSetting(IBuilderConfig* config) override{

    }

    void nvProfileSetting(IOptimizationProfile* profile) override {

    }
private:
    std::shared_ptr<ITRTFrameworkCreateEngineFromOnnxConfig> _engine_config{nullptr};
};

// class TRTEngineConfig_INT8 : public ITRTEngineConfig{
//     class INT8_Calibrator : public IInt8EntropyCalibrator{
//     public:
//         INT8_Calibrator(const int batch_size, const int input_height, const int input_width, const int input_channel) 
//                 : batch_size(batch_size), input_height(input_height), input_width(input_width), input_channel(input_channel){

//         }
//         int32_t getBatchSize() const noexcept override {
//             return batch_size;
//         }
//         bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override{
//             static int c = 0;
//             cudaMalloc(&bindings[0],input_height * input_width * input_channel * batch_size * sizeof(float));
//             return ++c % 10 == 0 ? false : true;
//         }
//         const void* readCalibrationCache(std::size_t& length) noexcept override{
//             return nullptr;
//         }
//         void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override{

//         }
//     private:
//         const int batch_size;
//         const int input_height, input_width, input_channel;
//     };
// public:
//     TRTEngineConfig_INT8(std::shared_ptr<ITRTEngineConfig> engine_config) 
//     : _engine_config(engine_config) {}

//     void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){
//         config->setFlag(nvinfer1::BuilderFlag::kINT8);
//         config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
//         config->setInt8Calibrator(new INT8_Calibrator(batch_size,input_height,input_width,input_channel));
//         if (model_config != NULL)
//             model_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
//     }

//     void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){
//         if (model_config != NULL)
//             model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
//     }

//     bool noDataTransfers(){
//         return model_config->noDataTransfers();
//     }
//     bool isDynamicBatchSize(){
//         return model_config->isDynamicBatchSize();
//     }
// };


class TRTEngineConfig_FP16 : public ITRTEngineConfig{
public:
    TRTEngineConfig_FP16(std::shared_ptr<ITRTEngineConfig> engine_config) 
                        : _engine_config(engine_config) {}

    void nvConfigSetting(IBuilderConfig* config) override {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        if (_engine_config != NULL)
            _engine_config->nvConfigSetting(config);
    }

    void nvProfileSetting(IOptimizationProfile* profile) override {
        if (_engine_config != NULL)
            _engine_config->nvProfileSetting(profile);
    }

    bool isDynamicBatchSize() override {
        return _engine_config->isDynamicBatchSize();
    }

};


// class TRTEngineConfig_DynamicBatch : public ITRTEngineConfig{
// public:
//     TRTEngineConfig_DynamicBatch(std::shared_ptr<ITRTEngineConfig> engine_config) 
//                                 : _engine_config(engine_config) {}
    
//     void nvConfigSetting(IBuilderConfig* config) override {
//         if (_engine_config != NULL)
//             _engine_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
//     }

//     void nvProfileSetting(IOptimizationProfile* profile) override {
//         profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kMIN, Dims4(1,input_channel,input_height,input_width));
//         profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kOPT, Dims4(batch_size,input_channel,input_height,input_width));
//         profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kMAX, Dims4(32,input_channel,input_height,input_width));
//         if (_engine_config != NULL)
//             _engine_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
//     }

//     bool isDynamicBatchSize() override {
//         return _engine_config->isDynamicBatchSize();
//     }
// };




} // TensorRT

} // ModelFramework