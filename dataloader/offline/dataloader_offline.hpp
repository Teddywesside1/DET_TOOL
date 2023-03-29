#pragma once

#include "dataloader.hpp"
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>
using namespace std;


class OfflineDataLoader : public IDataLoader{
public:
    OfflineDataLoader(const string dataset_dir, const int frame_total, const int round_total);

public:
    vector<shared_ptr<float>> getOneRound(const int round, const int batch_size, const int single_input_size, 
                                    std::function<void(shared_ptr<float>, cv::Mat&, const int)>) override;
    int getDatasetTotal() override;
    void start() override;
    void stop() override;
    bool isRunning() override;

private:

private:
    string dataset_dir = "";
    const int frame_total, round_total;
    atomic<bool> running_flag {false};
};


class OfflineDataLoaderFactory : public DataLoaderFactoryBase{
public:
    OfflineDataLoaderFactory(const string dataset_dir, const int frame_total, const int round_total) : 
                            dataset_dir(dataset_dir), frame_total(frame_total), round_total(round_total){

    }

    IDataLoader* createDataLoader() override {
        return new OfflineDataLoader(dataset_dir, frame_total, round_total);
    }

private:
    const string dataset_dir;
    const int frame_total, round_total;
    int round_count = 0;
};