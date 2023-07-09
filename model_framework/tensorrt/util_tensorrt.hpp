#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <exception>
#include <glog/logging.h>
#include <glog/log_severity.h>


namespace ModelFramework{

namespace TensorRT{

class TRTCreateEngineException : public std::exception{
public:
    TRTCreateEngineException(const std::string hint)
                            : _hint(hint){
    }
    const char* what() const noexcept {
        return _hint.c_str();
    }
private:
    std::string _hint;
};



class TensorrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override 
    {
        // suppress info-level messages
        if (severity == Severity::kINFO)
            LOG(INFO) << "tensorrt : " << msg;
        else if (severity == Severity::kERROR)
            LOG(ERROR) << "tensorrt : " << msg;
        else if (severity == Severity::kWARNING)
            LOG(WARNING) << "tensorrt : " << msg;
    }
};



void 
create_engine_from_onnx(const std::string &onnx_file_path);



} // TensorRT

} // ModelFramework
