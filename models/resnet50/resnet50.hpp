#pragma once


#include "model_runtime.hpp"
#include "model_instance.hpp"

class ResNet50 : public ModelRuntime{
public:
    ResNet50(const int batch_size, IModelInstance* model_instance);
public:
    void getInputHWC(int &input_height, int &input_width, int &input_channel);
    void getBatchsize(int &batchsize);
    // virtual void getDatasetTotal(int &dataset_total);
    // int inference(std::shared_ptr<float> data_ptr) override;
    // std::shared_ptr<float> getOneBatch(IDataLoader* dataloader_interface);

    int inferenceRounds(IDataLoader* dataloader_interface, const int round) override;
private:
    IModelInstance* model_instance = nullptr;
    int batch_size;
    int input_height, input_width, input_channel;

};


class ResNet50Factory : public ModelRuntimeFactoryBase{
public:
    ResNet50Factory(const int batch_size, IModelInstance* model_instance) :
                batch_size(batch_size), model_instance(model_instance){

    }
    
    ModelRuntime* createRuntimeModel(){
        return new ResNet50(batch_size,model_instance);
    }
private:
    int batch_size;
    IModelInstance* model_instance;
};