#pragma once


#include "interface_model_framework.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <exception>
#include "util_tensorrt.hpp"
#include "util_path.hpp"
#include "defs.hpp"


using namespace nvinfer1;

namespace ModelFramework{

namespace TensorRT{

class TRTRuntimeException : public std::exception {
public:
    TRTRuntimeException(const std::string hint)
                        : _hint(hint){

    }

    const char * what() const noexcept override {
        return _hint.c_str();
    }
private:
    std::string _hint;
};


class TRTModelFramework : public IModelFramework{
public:
    TRTModelFramework(const std::string model_path);
    /**
     * @brief make the framework inference, to fill the buffer with result data
     * 
     * @param batch_size should be equal to the batch_size of input data which has been inplaced in buffer
     */
    void framework_forward(const int batch_size) override;

    std::vector<void*>& get_buffer() override;

    ~TRTModelFramework();

private:
    void loadEngine(const std::string& engine_file_path);
    int getSizeByDim(const nvinfer1::Dims& dims);
    void prepareContext();
private:

    TensorrtLogger logger {};
    std::unique_ptr<ICudaEngine> engine{nullptr};
    std::unique_ptr<IExecutionContext> context{nullptr};
    std::vector<void*> buffers;

    bool _engine_built_with_implicit_batch = true;
};


} // TensorRT

} // ModelFramework




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