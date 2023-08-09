#pragma once


#include "interface_model_framework.hpp"
#include <memory>
#include <exception>
#include "util_path.hpp"
#include "defs.hpp"
#include <net.h>


namespace ModelFramework{

namespace NCNN{

class NCNNModelFramework : public IModelFramework {
public:
    NCNNModelFramework(const std::string model_param_path, const std::string model_bin_path);

    static void setMaxInputShape(const int maxBatch,
                                    const int maxHeight,
                                    const int maxWidth,
                                    const int maxChannel);

    /**
     * @brief 
     * 
     * @param batch_size 
     * @param input_height 
     * @param input_width 
     * @param input_channel 
     */
    void framework_forward(const int batch_size,
                            const int input_height,
                            const int input_width,
                            const int input_channel) override;

    /**
     * @brief Get the buffer object
     * 
     * @return std::vector<void*>& 
     */
    std::vector<void*>& get_buffer() override;

    ~NCNNModelFramework();
private:
    void loadNCNNModel(const std::string model_param_path,
                        const std::string model_bin_path);

    void prepareBuffer();
    
private:
    ncnn::Net _ncnn_model;
    ncnn::Extractor _ncnn_ex;
    std::vector<void*> _buffer;
    std::vector<ncnn::Mat> _output_blob_buffer;


private:
    static int maxBatch, maxHeight, maxWidth, maxChannel;
};

} // NCNN

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