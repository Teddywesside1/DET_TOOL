#include "container.hpp"
#include <iostream>
#include "logger.hpp"

using namespace std;

Container::Container(const string model_name, const int batch_size, const int round, 
                shared_ptr<ModelRuntimeFactoryBase> model_runtime_factory, shared_ptr<DataLoaderFactoryBase> dataloader_factory ) 
                : model_name(model_name), batch_size(batch_size), round(round) {
    m_model.reset(model_runtime_factory->createRuntimeModel());
    m_dataloader.reset(dataloader_factory->createDataLoader());
}



bool Container::start(){
    std::unique_lock<mutex> lck(m_locker);
    if (!m_model->isRunning() && !m_dataloader->isRunning()){
        logger << LogLevel::INFO << "trying to start model inference .. " << LOG_LINE_END;
        m_dataloader->start();
        logger << LogLevel::INFO << "successfully start dataloarder ! " << LOG_LINE_END;
        m_model->start(this,m_dataloader.get(),round);
        logger << LogLevel::INFO << "successfully start model_runtime ! " << LOG_LINE_END;
        return true;
    }else{
        return false;
    }
}


bool Container::stop(){
    std::unique_lock<mutex> lck(m_locker);
    if (m_model->isRunning() && m_dataloader->isRunning()){
        m_dataloader->stop();
        return true;
    }else{
        return false;
    }
    
}


bool Container::isRunning(){
    return m_model->isRunning() || m_dataloader->isRunning();
}

const string & Container::getName(){
    return model_name;
}

void Container::parseModelRuntimeStatus(int total, int cur, int FPS){
    logger << LogLevel::INFO << "progress : [" << cur << "/" << total << "] , FPS : " << FPS << LOG_LINE_END;
    this->total = total;
    this->cur = cur;
    this->FPS = FPS;
}


void Container::getProgress(int &total, int &cur, int &FPS, int &batch_size){
    total = this->total;
    cur = this->cur;
    FPS = this->FPS;
    batch_size = this->batch_size;
}


void Container::resetProgress(){
    this->total = 0;
    this->cur = 0;
    this->FPS = 0;
}