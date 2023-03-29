#pragma once

#include "dataloader.hpp"
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

using namespace std;

class FakelineDataLoader : public IDataLoader{
public:
    FakelineDataLoader(const string dataset_dir, const int frame_total);

public:
    vector<shared_ptr<float>> getOneRound(const int round, const int batch_size, const int single_input_size) override;
    int getDatasetTotal() override;
    void start() override;
    void stop() override;
    bool isRunning() override;

private:

private:
    const string dataset_dir;
    const int frame_total;
    // bool running_flag = false, stop_flag = false;
    mutex locker;
    atomic<bool> running_flag {false};

    shared_ptr<float> data_ptr {nullptr};
    vector<shared_ptr<float>> roundsPtr;
    int frame_count = 0;
};


class FakeDataLoaderFactory : public DataLoaderFactoryBase{
public:
    FakeDataLoaderFactory(const string dataset_dir, const int frame_total) : 
                            dataset_dir(dataset_dir), frame_total(frame_total){

    }

    IDataLoader* createDataLoader() override {
        return new FakelineDataLoader(dataset_dir, frame_total);
    }

private:
    const string dataset_dir;
    const int frame_total;
};