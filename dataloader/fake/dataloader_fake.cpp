#include "dataloader_fake.hpp"
#include <iostream>
using namespace std;

FakelineDataLoader::FakelineDataLoader(const string dataset_dir, const int frame_total) 
                        : dataset_dir(dataset_dir), frame_total(frame_total){
                     
}

vector<shared_ptr<float>> FakelineDataLoader::getOneRound(const int round, const int batch_size, const int single_input_size){
    if (data_ptr == nullptr)
        data_ptr.reset(new float[batch_size * single_input_size], [](float* ptr){delete ptr;});

    if (roundsPtr.size() == 0){
        for (int i = 0 ; i < round ; ++ i){
            roundsPtr.push_back(data_ptr);
        }
    }

    if (running_flag.load() == false || frame_count >= frame_total){
        running_flag.store(false);
        frame_count = 0;
        data_ptr.reset();
        return vector<shared_ptr<float>>();
    }
    
    frame_count += round * batch_size;
    
    return roundsPtr;
}


int FakelineDataLoader::getDatasetTotal(){
    return frame_total;
}


void FakelineDataLoader::start() {
    running_flag = true;
}

void FakelineDataLoader::stop() {
    running_flag.store(false);
}

bool FakelineDataLoader::isRunning() {
    return running_flag.load();
}