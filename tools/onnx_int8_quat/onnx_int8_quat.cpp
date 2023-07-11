#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <exception>
#include <glog/logging.h>
#include <glog/log_severity.h>
#include <fstream>
#include <string>
#include <climits>
#include <opencv2/opencv.hpp>

#include "util_tensorrt.hpp"
#include "dataloader_objdet_2d.hpp"
#include "common.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"


using namespace nvinfer1;


static void show_usage(void){
    LOG(ERROR) << "usage `onnx_int8_quat"
                                    << " --onnx=/path/to/onnx_to_load"
                                    << " --saveEngine=/path/to/engine_to_save"
                                    << " --minBatch=min"
                                    << " --optBatch=opt"
                                    << " --maxBatch=max`"
                << "\n `minBatch`, `optBatch`, `maxBatch` shoulde be provided when the onnx model is built with implicit batch_size";
}


class INT8_Calibrator : public IInt8EntropyCalibrator{
public:
    INT8_Calibrator(std::shared_ptr<INetworkDefinition> network)
                : _network(network){
        if ()
    }
    int32_t getBatchSize() const noexcept override {
        return _batch_size;
    }
    bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override{
        auto resize_strategy = 
            [&](const cv::Mat& image){
                const int ori_height = image.rows;
                const int ori_width = image.cols;
                int fix_height, fix_width;
                float scale;
                if (ori_height > ori_width){
                    fix_height = _input_height;
                    scale = _input_height / static_cast<float>(ori_height);
                    fix_width = static_cast<int>(ori_width * scale);
                }else{
                    fix_width = _input_width;
                    scale = _input_width / static_cast<float>(ori_width);
                    fix_height = static_cast<int>(ori_height * scale);
                }

                cv::Mat resized_image;
                cv::resize(image, resized_image, {fix_width, fix_height});
                auto out = std::make_shared<cv::Mat>(_input_height, _input_width, CV_8UC3, cv::Scalar{0, 0, 0});
                resized_image.copyTo((*out)(cv::Rect(0, 0, fix_width, fix_height)));
                return out;
            };
        const int single_channel_pixel_size = _input_height * _input_width;
        const int single_image_float_element_size = single_channel_pixel_size * _input_channel;
        auto flatten_strategy = 
            [&](float* data_ptr, const cv::Mat& image){
                int idx = 0;
                for (int r = 0 ; r < _input_height ; ++ r){
                    uchar* pixel_ptr = image.data + r * image.step;
                    for (int c = 0 ; c < _input_width ; ++ c){
                        data_ptr[idx] = pixel_ptr[2] / 255.0;
                        data_ptr[idx + single_channel_pixel_size] = pixel_ptr[1] / 255.0;
                        data_ptr[idx + 2 * single_channel_pixel_size] = pixel_ptr[0] / 255.0;
                        pixel_ptr += 3;
                        ++ idx;}
                }
                return single_image_float_element_size;
            };
        
        auto image_ptr = std::make_shared<cv::Mat>(cv::imread("/data/binfeng/projects/server_multi-platform/images/bus.jpg"));

        DataLoader::DataLoaderObjDet2D dataloader;
        for (int i = 0 ; i < _batch_size ; ++ i)
            dataloader.push(image_ptr);
        
        for (int i = 0 ; i < nbBindings ; ++ i){
            if (_network->getInput(i)->isNetworkInput()){
                dataloader.
            }
        }

        static int c = 0;
        cudaMalloc(&bindings[0],input_height * input_width * input_channel * batch_size * sizeof(float));
        return ++c % 10 == 0 ? false : true;
    }
    const void* readCalibrationCache(std::size_t& length) noexcept override{
        return nullptr;
    }
    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override{

    }
private:
    const int _batch_size = 4;
    int _input_height, _intput_width, _input_channel;
    std::shared_ptr<INetworkDefinition> _network {};
    std::vector<void *> _buffers;
};


static std::shared_ptr<IHostMemory> parse_onnx_file(
                                            const std::string& model_path,
                                            const int minBatch,
                                            const int optBatch,
                                            const int maxBatch){
    ModelFramework::TensorRT::TensorrtLogger logger;
    std::unique_ptr<IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    std::shared_ptr<INetworkDefinition> network{builder->createNetworkV2(1)};
    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {   
        LOG(ERROR) << ("parse onnx file failed !");
    }

    // if onnx is built with implicit batch_size, make profile for config
    if (network->getInput(0)->getDimensions().d[0] == -1){
        if (minBatch == -1 || optBatch == -1 || maxBatch == -1){
            show_usage();
            exit(EXIT_FAILURE);
        }
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (int i = 0 ; i < network->getNbInputs() ; ++ i){
            Dims dims = network->getInput(i)->getDimensions();
            profile->setDimensions(network->getInput(i)->getName(), OptProfileSelector::kMIN, Dims4(minBatch,dims.d[1],dims.d[2],dims.d[3]));
            profile->setDimensions(network->getInput(i)->getName(), OptProfileSelector::kOPT, Dims4(optBatch,dims.d[1],dims.d[2],dims.d[3]));
            profile->setDimensions(network->getInput(i)->getName(), OptProfileSelector::kMAX, Dims4(maxBatch,dims.d[1],dims.d[2],dims.d[3]));    
        }
        config->addOptimizationProfile(profile);
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    config->setInt8Calibrator(new INT8_Calibrator(batch_size,input_height,input_width,input_channel));
    model_config->nvConfigSetting(config.get(), batch_size, input_height, input_width, input_channel);
    generate TensorRT engine optimized for the target platform
    builder->setMaxBatchSize(32);
    std::shared_ptr<nvinfer1::IHostMemory> engineOutput;
    engineOutput.reset(builder->buildSerializedNetwork(*network, *config));
    return engineOutput;
}

static void saveEngine(const std::string& outputFilePath, std::shared_ptr<nvinfer1::IHostMemory> engine_serialized)
{
    /*  Write to disk   */    
    std::ofstream outputFile;
    outputFile.open(outputFilePath, std::ios::out | std::ios::binary);
    outputFile.write((char*)engine_serialized->data(), engine_serialized->size());
    if(!outputFile.good()){
        LOG(ERROR) << "save engine failed !";
    }
    outputFile.close();
}


// void 
// create_int8_engine_from_onnx(const std::string &onnx_file_path){
//     TensorrtLogger logger;
//     saveEngine(onnx_file_path + ".engine", 
//                 parse_onnx_file(onnx_file_path, logger));
// }







int main(int argc, char **argv){
    sample::Arguments args = sample::argsToArgumentsMap(argc, argv);

    if (sample::parseHelp(args)){
        sample::AllOptions::help(std::cout);
        return EXIT_SUCCESS;
    }

    if (args.find("--onnx") == args.end()
    || args.find("--saveEngine") == args.end()){
        show_usage();
        return EXIT_FAILURE;
    }

    const std::string onnx_file_path = args.find("--onnx")->second;
    const std::string save_engine_path = args.find("--saveEngine")->second;
    int minBatch = -1;
    int optBatch = -1;
    int maxBatch = -1;
    try{
        if (args.find("--minBatch") != args.end())
            minBatch = std::stoi(args.find("--minBatch")->second);
        if (args.find("--optBatch") != args.end())
            optBatch = std::stoi(args.find("--optBatch")->second);
        if (args.find("--maxBatch") != args.end())
            maxBatch = std::stoi(args.find("--maxBatch")->second);
    }catch(std::exception e){

    }
    parse_onnx_file(onnx_file_path,
                    minBatch,
                    optBatch,
                    maxBatch);

    return 0;
}