#pragma once

#include <mutex>
#include <memory>
#include "dataloader.hpp"
#include "model_instance.hpp"
#include "model_runtime.hpp"

using namespace std;

class Container : public IModelObserve{
public:
    Container(const string model_name, const int batch_size, const int round, shared_ptr<ModelRuntimeFactoryBase> model_runtime_factory,
            shared_ptr<DataLoaderFactoryBase> dataloader_factory );
    bool start();
    bool stop();
    void getProgress(int &total, int &cur, int &FPS, int &batch_size);
    void resetProgress();
    bool isRunning();

    const string& getName();

private:
    void parseModelRuntimeStatus(int total, int cur, int FPS) override;

private:
    unique_ptr<ModelRuntime> m_model{nullptr};
    unique_ptr<IDataLoader> m_dataloader{nullptr};
    std::mutex m_locker;

    const string model_name;
    int total = 0, cur = 0, FPS = 0;
    const int batch_size, round;
};