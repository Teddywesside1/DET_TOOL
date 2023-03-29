#pragma once
#include "CMakeConfig.h"
#include <memory>
#include <vector>

#if (WITHOUT_IMAGE_READING)

#else
    #include <opencv2/opencv.hpp>
#endif

using namespace std;

class IDataLoader{
public:
#if (WITHOUT_IMAGE_READING)
    virtual vector<shared_ptr<float>> getOneRound(const int round, const int batch_size, const int single_input_size) = 0;
#else
    virtual vector<shared_ptr<float>> getOneRound(const int round, const int batch_size, const int single_input_size, std::function<void(std::shared_ptr<float>, cv::Mat&, const int)>) = 0;
#endif
    virtual int getDatasetTotal() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool isRunning() = 0;
};


class DataLoaderFactoryBase{
public:
    virtual IDataLoader* createDataLoader() = 0;
};