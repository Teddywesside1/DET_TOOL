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
#include <dirent.h>

#include "util_tensorrt.hpp"
#include "dataloader_objdet_2d.hpp"
#include "common.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "resize_strategy.hpp"
#include "flatten_strategy.hpp"

#include "util_data_structure.hpp"


using namespace nvinfer1;

using namespace ModelInference::ProcessStrategy;
using namespace DataLoader;


#define MAX_CALIB_IMAGE_COUNT 5000


static void show_usage(void){
    LOG(ERROR) << "usage \n`onnx_int8_quat"
                                    << "\n--onnx=/path/to/onnx_to_load"
                                    << "\n--saveEngine=/path/to/engine_to_save"
                                    << "\n--calib_dataset_dir=/path/to/calib_dataset_dir"
                                    << "\n--use_image_count=number"
                                    << "\n--minBatch=min"
                                    << "\n--optBatch=opt"
                                    << "\n--maxBatch=max`"
                << "\n `minBatch`, `optBatch`, `maxBatch` shoulde be provided when the onnx model is built with implicit batch_size";
}


class INT8_Calibrator : public IInt8EntropyCalibrator2{
public:
    INT8_Calibrator(std::shared_ptr<INetworkDefinition> network,
                    const int batch_size,
                    const std::string& calib_dataset_dir,
                    const int use_image_count)
                : _network(network)
                , _batch_size(batch_size)
                , _input_height(_network->getInput(0)->getDimensions().d[2])
                , _input_width(_network->getInput(0)->getDimensions().d[3])
                , _input_channel(_network->getInput(0)->getDimensions().d[1])
                , _resize_strategy(_input_height, _input_width)
                , _block_dataloader(_batch_size * 2){
        const int nbBindings = _network->getNbInputs() + _network->getNbOutputs();
        _buffers.resize(nbBindings);
        LOG(INFO) << "start building input binding memory ... ";
        for (int i = 0 ; i < _network->getNbInputs() ; ++ i){
            Dims dim = _network->getInput(i)->getDimensions();
            int byte_size_to_malloc = _batch_size * sizeof(float);
            for (int j = 1 ; j < dim.nbDims ; ++ j){
                byte_size_to_malloc *= dim.d[j];
            }
            cudaMallocManaged(&_buffers[i], byte_size_to_malloc);

            cudaDeviceSynchronize();
        }
        LOG(INFO) << "start building output binding memory ... ";
        for (int i = 0 ; i < _network->getNbOutputs() ; ++ i){
            Dims dim = _network->getOutput(i)->getDimensions();
            int byte_size_to_malloc = _batch_size * sizeof(float);
            for (int j = 1 ; j < dim.nbDims ; ++ j){
                byte_size_to_malloc *= dim.d[j];
            }
            cudaMallocManaged(&_buffers[_network->getNbInputs() + i], byte_size_to_malloc);
            cudaDeviceSynchronize();
        }
        LOG(INFO) << "start image reading thread ...";
        _img_reading_thread = std::thread(&INT8_Calibrator::img_reading_thread_entry, 
                                            this, 
                                            calib_dataset_dir,
                                            MAX_CALIB_IMAGE_COUNT);
        _img_reading_thread.detach();
    }
    int32_t getBatchSize() const noexcept override {
        return _batch_size;
    }
    bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override{
        CHECK(_buffers.size() == nbBindings);
        for (int i = 0 ; i < nbBindings ; ++ i){
            bindings[i] = static_cast<void*>(_buffers[i]);
        }
        // LOG(INFO) << "get one batch";
        std::vector<float*> input_ptrs;
        for (int i = 0 ; i < _network->getNbInputs() ; ++ i){
            input_ptrs.push_back(static_cast<float*>(_buffers[i]));
        }

        std::vector<BatchInfo> _;
        bool ret = _block_dataloader.get_explicit_batch(_resize_strategy, 
                                                        _flatten_strategy, 
                                                        input_ptrs, 
                                                        _ , 
                                                        _batch_size);
        
        return ret;

        // return true;
        
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override{
        return nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override{

    }

    ~INT8_Calibrator(){
        for (int i = 0 ; i < _buffers.size() ; ++ i){
            cudaFree(_buffers[i]);
        }
        _buffers.clear();
    }

private:
    void img_reading_thread_entry(const std::string dataset_dir, const int max_use_image_count){
        DIR *pDir;
        struct dirent *dir_ptr;
        if (!(pDir = opendir(dataset_dir.c_str()))){
            throw std::runtime_error("INT8_Calibrator, img_reading : dataset_dir not exists! dir : [" + dataset_dir + "] failed !" );
        }
        std::vector<std::string> files;
        while ((dir_ptr = readdir(pDir)) != 0){
            std::string tmp(dir_ptr->d_name);
            int idx = tmp.find_last_of('.');
            std::string suffix = tmp.substr(idx + 1, tmp.size() - idx - 1);
            if ("jpg" != suffix && "png" != suffix) continue;
            files.push_back(dataset_dir + "/" + tmp);
        }
        closedir(pDir);
        const int total = files.size();
        const int image_to_use = std::min(max_use_image_count, total);
        LOG(INFO) << "start reading images from dir [" << dataset_dir << "], total count : " << total;
        int count = 0;
        for (const std::string & file_path : files){
            const auto img_ptr = std::make_shared<cv::Mat>(cv::imread(file_path));
            _block_dataloader.push(img_ptr);
            ++ count;
            LOG(INFO) << "dataloader : [" << count << " / " << image_to_use << "]";
            if (count == image_to_use) break;
        }
        LOG(INFO) << "image reading thread finished job, calling block queue stop ..";
        _block_dataloader.callStop();
        
    }

private:
    std::shared_ptr<INetworkDefinition> _network {};
    const int _batch_size = 4;
    const int _input_height;
    const int _input_width;
    const int _input_channel;
    ResizeStrategyPadding _resize_strategy;
    FlattenStrategyHWC2CHW _flatten_strategy;
    
    std::vector<void *> _buffers;

    std::thread _img_reading_thread;
    BlockDataLoaderObjDet2D _block_dataloader;
};


static std::shared_ptr<IHostMemory> parse_onnx_file(
                                        const std::string& model_path,
                                        const std::string& calib_dataset_dir,
                                        const int use_image_count,
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
        exit(EXIT_FAILURE);
    }

    LOG(INFO) << "finish parsing onnx file, starting generate engine !";

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
    // set `kINT8` building config
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    // get INT8 calibrator
    auto calibrator = std::make_shared<INT8_Calibrator>(network,
                                                    4, 
                                                    calib_dataset_dir,
                                                    use_image_count);
    config->setInt8Calibrator(calibrator.get());
    // generate TensorRT engine optimized for the target platform
    // builder->setMaxBatchSize(32);
    std::shared_ptr<nvinfer1::IHostMemory> engineOutput;
    LOG(INFO) << "starting serializing engine ! ";
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



int main(int argc, char **argv){
    sample::Arguments args = sample::argsToArgumentsMap(argc, argv);

    if (sample::parseHelp(args)){
        sample::AllOptions::help(std::cout);
        return EXIT_SUCCESS;
    }

    if (args.find("--onnx") == args.end()
    || args.find("--saveEngine") == args.end()
    || args.find("--calib_dataset_dir") == args.end()){
        show_usage();
        return EXIT_FAILURE;
    }

    const std::string onnx_file_path = args.find("--onnx")->second;
    const std::string save_engine_path = args.find("--saveEngine")->second;
    const std::string calib_dataset_dir = args.find("--calib_dataset_dir")->second;
    int minBatch = -1;
    int optBatch = -1;
    int maxBatch = -1;
    int use_image_count = MAX_CALIB_IMAGE_COUNT;
    try{
        if (args.find("--minBatch") != args.end())
            minBatch = std::stoi(args.find("--minBatch")->second);
        if (args.find("--optBatch") != args.end())
            optBatch = std::stoi(args.find("--optBatch")->second);
        if (args.find("--maxBatch") != args.end())
            maxBatch = std::stoi(args.find("--maxBatch")->second);
        if (args.find("--use_image_count") != args.end())
            use_image_count = std::stoi(args.find("--use_image_count")->second);
    }catch(std::exception e){
        show_usage();
        return EXIT_FAILURE;
    }
    auto onnx2engine = parse_onnx_file(onnx_file_path,
                                    calib_dataset_dir,
                                    use_image_count,
                                    minBatch,
                                    optBatch,
                                    maxBatch);
    LOG(INFO) << "saving engine to : " << save_engine_path;
    saveEngine(save_engine_path, onnx2engine);
    return 0;
}