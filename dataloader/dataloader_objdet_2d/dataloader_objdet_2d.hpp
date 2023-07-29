#pragma once

#include "interface_dataloader.hpp"
#include "util_data_structure.hpp"
#include <queue>
#include "defs.hpp"

namespace DataLoader{

class DataLoaderObjDet2D : public IDataLoader{
public:
    DataLoaderObjDet2D();
    /**
     * @brief Get the batch object
     * 
     * @param resize_strategy 
     * @param flatten_strategy 
     * @param data_ptrs 
     * @param batch_info 
     * @return int 
     */
    int get_buffer_as_one_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info) override;
    
    virtual bool
    get_explicit_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info,
            const int batch_size) override;

    bool push(const std::shared_ptr<cv::Mat> & image) override;


private:
    std::queue<std::shared_ptr<cv::Mat>> _image_buffer;
};





class BlockDataLoaderObjDet2D : public IDataLoader{
public:
    BlockDataLoaderObjDet2D(const int maxBlockQueueSize);
    /**
     * @brief Get the batch object
     * 
     * @param resize_strategy 
     * @param flatten_strategy 
     * @param data_ptrs 
     * @param batch_info 
     * @return int 
     */
    int get_buffer_as_one_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info) override;

    virtual bool
    get_explicit_batch(ModelInference::ProcessStrategy::IResizeStrategy &resize_strategy,
            ModelInference::ProcessStrategy::IFlattenStrategy &flatten_strategy,
            std::vector<float*> & data_ptrs,
            std::vector<BatchInfo> & batch_info,
            const int batch_size) override;

    bool push(const std::shared_ptr<cv::Mat> & image) override;


    void callStop(){
        _image_buffer.callStop();
    }


private:
    BlockQueue<std::shared_ptr<cv::Mat>> _image_buffer;
};




} // DataLoader


