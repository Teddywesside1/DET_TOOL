#include "model_runtime.hpp"
#include <ctime>

using namespace std;


void ModelRuntime::start(IModelObserve* model_interface, IDataLoader* dataloader_interface, const int round){
    if (!m_runningThread.joinable())
        m_runningThread = std::thread(&ModelRuntime::threadRun,this,model_interface, dataloader_interface, round);    
}

bool ModelRuntime::isRunning(){
    return m_runningThread.joinable();
}


void ModelRuntime::threadRun(IModelObserve* model_interface, IDataLoader* dataloader_interface, const int round){
    int dataset_total = dataloader_interface->getDatasetTotal();
    int frame_count = 0;
    int64_t cum_duration = 0;

    while (true){   
        // auto data_ptr = getOneBatch(dataloader_interface);
        // if (!data_ptr) break;

        // auto start = chrono::high_resolution_clock::now();
        // frame_count += inference(data_ptr);
        // auto now = chrono::high_resolution_clock::now();

        auto start = chrono::high_resolution_clock::now();
        int ret = inferenceRounds(dataloader_interface, round);
        auto now = chrono::high_resolution_clock::now();
        if (ret == -1) break;
        
        auto duration = chrono::duration_cast<chrono::microseconds>(now - start).count();
        cum_duration += duration;
        frame_count += ret;
        int FPS = (double)frame_count / ((double)cum_duration / 1e6);
        model_interface->parseModelRuntimeStatus(dataset_total, frame_count, FPS);
    }
    dataloader_interface->stop();
    m_runningThread.detach();
}


ModelRuntime::~ModelRuntime(){

}