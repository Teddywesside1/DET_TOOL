#pragma once
#include "dataloader.hpp"
#include "block_queue.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
using namespace std;


class OnlineDataLoader : public IDataLoader{
public:
    OnlineDataLoader();
    OnlineDataLoader(const string dataset_dir);

public:
    vector<shared_ptr<float>> getOneRound(const int round, const int batch_size, const int single_input_size, 
                                    std::function<void(shared_ptr<float>, cv::Mat&, const int)>) override;
    int getDatasetTotal() override;
    void start() override;
    void stop() override;
    bool isRunning() override;

private:
    void threadRun();

private:
    string dataset_dir = "";
    shared_ptr<BlockQueueInterface> blockqueue_interface = make_shared<BlockQueue>();
    std::thread m_runningThread;
    int buffer_read_size = 100;
};


class OnlineDataLoaderFactory : public DataLoaderFactoryBase{
public:
    OnlineDataLoaderFactory(const string dataset_dir) : 
                            dataset_dir(dataset_dir){

    }

    IDataLoader* createDataLoader() override {
        return new OnlineDataLoader(dataset_dir);
    }

private:
    const string dataset_dir;
};