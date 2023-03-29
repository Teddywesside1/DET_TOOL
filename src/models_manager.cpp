#include "models_manager.hpp"
#include <iostream>

using namespace std;

Manager::Manager(){

}

bool Manager::startModel(const string& model_name){
    if (isRunning()) return false;

    if (table.find(model_name) == table.end()){
        return false;
    }else{
        shared_ptr<Container> model_ptr = table[model_name];
        
        if (model_ptr->start()){
            running_model = model_ptr;
            return true;
        }else{
            return false;
        }
    }
}

bool Manager::stopModel(){
    if (!isRunning()) return false;

    return running_model->stop();
}

bool Manager::stopModel(const string model_name){
    if (!isRunning()) return false;

    auto model_ptr = table[model_name];
    if (model_ptr == nullptr || !model_ptr->isRunning()) return false;

    return model_ptr->stop();
}

bool Manager::getProgress(const string model_name, int &total, int &cur, int &FPS, int &batch_size){
    auto model_ptr = table[model_name];
    if (model_ptr == nullptr) return false;
    model_ptr->getProgress(total,cur,FPS,batch_size);
    return true;
    
}

bool Manager::resetProgress(const string model_name){
    auto model_ptr = table[model_name];
    if (model_ptr == nullptr) return false;
    else if (model_ptr->isRunning()) return false;

    model_ptr->resetProgress();

    return true;
}


bool Manager::createModel(const string& model_name, shared_ptr<Container> container_ptr){
    table[model_name] = container_ptr;
    return true;
}


void Manager::getAllModelsInfo(vector<ModelInfo> &models_progress){
    int total, cur, FPS, batch_size;
    for (auto& p : table){
        p.second->getProgress(total,cur,FPS,batch_size);
        models_progress.emplace_back(p.first, p.second->isRunning(), total, cur, FPS, batch_size);
    }
}


bool Manager::isRunning(){
    return running_model != nullptr && running_model->isRunning();
}



const string Manager::getRunningModelName(){
    if (running_model == nullptr) 
        return "";
    else 
        return running_model->getName();
}