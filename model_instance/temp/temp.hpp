#pragma once

#include "model_instance.hpp"

using namespace std;

class TempModelInstance : public IModelInstance{
public:
    TempModelInstance(const int delay, const int batch_size);
    // int inference(std::shared_ptr<void> data_ptr) override;
    void inferenceRounds(vector<shared_ptr<float>>& data_ptr) override;
    ~TempModelInstance(){}

private:
    const int delay;
    const int batch_size;
};

