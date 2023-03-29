#pragma once
#include <thread>
#include <memory>
#include "model_instance.hpp"
#include "dataloader.hpp"




class IModelObserve{
public:
    virtual void parseModelRuntimeStatus(int total, int cur, int FPS) = 0;
};


class ModelRuntime{
public:
    void start(IModelObserve* model_interface, IDataLoader* dataloader_interface, const int round);
    bool isRunning();
    
    virtual ~ModelRuntime();
protected:
    // virtual int inference(std::shared_ptr<float> data_ptr) = 0;
    // virtual std::shared_ptr<float> getOneBatch(IDataLoader* dataloader_interface) = 0;

    virtual int inferenceRounds(IDataLoader* dataloader_interface, const int round) = 0;
private:
    // void warmUp();
    void threadRun(IModelObserve* model_interface, IDataLoader* dataloader_interface, const int round);

private:
    std::thread m_runningThread;
};

class ModelRuntimeFactoryBase{
public:
    virtual ModelRuntime* createRuntimeModel() = 0;
};
