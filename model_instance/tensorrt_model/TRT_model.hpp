#pragma once


#include "model_instance.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <iostream>

using namespace std;
using namespace nvinfer1;

class ITRTModelConfig {
public:
    virtual void nvProfileSetting(IOptimizationProfile* profile, const string& input_blob_name, const int batch_size, const int input_height, const int input_width, const int input_channel) = 0;
    virtual void nvConfigSetting(IBuilderConfig* config, const int batch_size, const int input_height, const int input_width, const int input_channel) = 0;
    virtual bool noDataTransfers() = 0;
    virtual bool isDynamicBatchSize() = 0;
protected:
    ITRTModelConfig(shared_ptr<ITRTModelConfig> model_config) : model_config (model_config) {}
    shared_ptr<ITRTModelConfig> model_config;
};


class TRTModelInstance : public IModelInstance{
    class logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override 
        {
            // suppress info-level messages
            if (severity == Severity::kINFO)
                cout << " --  [I] : " << msg << endl;
            else if (severity == Severity::kERROR)
                cout << " --  [E] : " << msg << endl;
            else if (severity == Severity::kWARNING)
                cout << " --  [W] : " << msg << endl;
        }
    };
public:
    TRTModelInstance(const string model_path, const string input_blob_name, const int batch_size,
                const int input_height, const int input_width, const int input_channel, shared_ptr<ITRTModelConfig> model_config);
    // int inference(std::shared_ptr<void> data_ptr) override;
    void inferenceRounds(vector<shared_ptr<float>>& data_ptr) override;
    ~TRTModelInstance(){}

private:
    void analyseModelEngine();
    void parseOnnxModel(const string& model_path);
    void saveEngine(const string& output_file_path, unique_ptr<nvinfer1::IHostMemory>& engine_output);
    void loadEngine(const string& engine_file_path);
    int getSizeByDim(const nvinfer1::Dims& dims);
    void prepareContext();
private:
    const int batch_size;
    int input_height, input_width, input_channel;
    const string input_blob_name;

    logger tlogger;
    unique_ptr<ICudaEngine> engine{nullptr};
    unique_ptr<IExecutionContext> context{nullptr};
    cudaStream_t stream;
    vector<void*> buffers;
    shared_ptr<ITRTModelConfig> model_config;
};


// class TRTModelInstance_Factory : public ModelInstanceFactoryBase{
// public:
//     TRTModelInstance_Factory(const string model_path, const string input_blob_name, const int batch_size,
//                 const int input_height, const int input_width, const int input_channel = 3)
//                 : model_path(model_path), input_blob_name(input_blob_name), batch_size(batch_size), 
//                 input_height(input_height), input_width(input_width), input_channel(input_channel){

//     }

//     IModelInstance* createModelInstance(){
//         return new TRTModelInstance_DynamicBS(model_path,input_blob_name,batch_size,input_height,input_width,input_channel);
//     }
// private:
//     const string model_path;
//     const int batch_size;
//     int input_height, input_width, input_channel;
//     const string input_blob_name;
// };