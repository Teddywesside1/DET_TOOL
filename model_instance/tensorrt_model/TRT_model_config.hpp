#include "TRT_model.hpp"

class TRTModelConfig_Default : public ITRTModelConfig{
public:
    TRTModelConfig_Default(shared_ptr<ITRTModelConfig> model_ptr) : ITRTModelConfig(model_ptr) {}

    void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){

    }

    void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){

    }

    bool noDataTransfers(){
        return false;
    }
    bool isDynamicBatchSize(){
        return false;
    }
};

class TRTModelConfig_INT8 : public ITRTModelConfig{
    class INT8_Calibrator : public IInt8EntropyCalibrator{
    public:
        INT8_Calibrator(const int batch_size, const int input_height, const int input_width, const int input_channel) 
                : batch_size(batch_size), input_height(input_height), input_width(input_width), input_channel(input_channel){

        }
        int32_t getBatchSize() const noexcept override {
            return batch_size;
        }
        bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override{
            static int c = 0;
            cout << "getBatch() nbBingdings : " << nbBindings << endl;
            for (int i = 0 ; i < nbBindings ; ++ i){
                cout << names[i] << endl;
            }
            cudaMalloc(&bindings[0],input_height * input_width * input_channel * batch_size * sizeof(float));
            return ++c % 10 == 0 ? false : true;
        }
        const void* readCalibrationCache(std::size_t& length) noexcept override{
            return nullptr;
        }
        void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override{

        }
    private:
        const int batch_size;
        const int input_height, input_width, input_channel;
    };
public:
    TRTModelConfig_INT8(shared_ptr<ITRTModelConfig> model_ptr) : ITRTModelConfig(model_ptr) {}

    void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        config->setInt8Calibrator(new INT8_Calibrator(batch_size,input_height,input_width,input_channel));
        if (model_config != NULL)
            model_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
    }

    void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){
        if (model_config != NULL)
            model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    }

    bool noDataTransfers(){
        return model_config->noDataTransfers();
    }
    bool isDynamicBatchSize(){
        return model_config->isDynamicBatchSize();
    }
};


class TRTModelConfig_FP16 : public ITRTModelConfig{
public:
    TRTModelConfig_FP16(shared_ptr<ITRTModelConfig> model_ptr) : ITRTModelConfig(model_ptr) {}
    void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        if (model_config != NULL)
            model_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
    }

    void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){
        if (model_config != NULL)
            model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    }

    bool noDataTransfers(){
        return model_config->noDataTransfers();
    }
    bool isDynamicBatchSize(){
        return model_config->isDynamicBatchSize();
    }

};


class TRTModelConfig_DynamicBatch : public ITRTModelConfig{
public:
    TRTModelConfig_DynamicBatch(shared_ptr<ITRTModelConfig> model_ptr) : ITRTModelConfig(model_ptr) {}
    void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){
        if (model_config != NULL)
            model_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
    }

    void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){
        profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kMIN, Dims4(1,input_channel,input_height,input_width));
        profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kOPT, Dims4(batch_size,input_channel,input_height,input_width));
        profile->setDimensions(input_blob_name.c_str(), OptProfileSelector::kMAX, Dims4(32,input_channel,input_height,input_width));
        if (model_config != NULL)
            model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    }

    bool noDataTransfers(){
        return model_config->noDataTransfers();
    }
    bool isDynamicBatchSize(){
        return true;
    }
};


class TRTModelConfig_NoDataTransfers : public ITRTModelConfig{
public:
    TRTModelConfig_NoDataTransfers(shared_ptr<ITRTModelConfig> model_ptr) : ITRTModelConfig(model_ptr) {}
    void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel){
        if (model_config != NULL)
            model_config->nvConfigSetting(config, batch_size, input_height, input_width, input_channel);
    }

    void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel){
        if (model_config != NULL)
            model_config->nvProfileSetting(profile, input_blob_name, batch_size, input_height, input_width, input_channel);
    }

    bool noDataTransfers(){
        return true;
    }
    bool isDynamicBatchSize(){
        return model_config->isDynamicBatchSize();
    }

};