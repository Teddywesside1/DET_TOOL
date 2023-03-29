#pragma once

#include <memory>
#include <vector>



class IModelInstance{
public:
    virtual void inferenceRounds(std::vector<std::shared_ptr<float>>& data_ptr) = 0;
    virtual ~IModelInstance(){}
};

class ModelInstanceFactoryBase{
public:
    virtual IModelInstance* createModelInstance() = 0;
};