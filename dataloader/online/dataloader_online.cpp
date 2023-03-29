#include "dataloader_online.hpp"
#include <dirent.h>
#include "common.hpp"

using namespace std;


OnlineDataLoader::OnlineDataLoader() : dataset_dir(""){ 

}

OnlineDataLoader::OnlineDataLoader(const string dataset_dir) : dataset_dir(dataset_dir){

}


vector<shared_ptr<float>> OnlineDataLoader::getOneRound(const int batch_count_one_round, const int batch_size, const int single_input_size, 
                                                        std::function<void(shared_ptr<float>, cv::Mat&, const int)> place_method){
    vector<shared_ptr<float>> ret;
    
    for (int i = 0 ; i < batch_count_one_round ; ++ i){
        shared_ptr<float> data_ptr {new float[single_input_size * batch_size], [](float* ptr){delete ptr;}};
        for (int j = 0 ; j < batch_size ; ++ j){
            shared_ptr<cv::Mat> image_ptr = blockqueue_interface->take();
            if (image_ptr == NULL){
                return vector<shared_ptr<float>>();
            }
            cv::Mat& image = *image_ptr;
            place_method(data_ptr, image, j);
        }
        ret.push_back(data_ptr);
    }

    return ret;
}



int OnlineDataLoader::getDatasetTotal(){
    DIR *pDir;
    struct dirent *dir_ptr;
    if (!(pDir = opendir(dataset_dir.c_str()))){
        throw runtime_error("dataset_dir not exists! dir : [" + dataset_dir + "] failed !" );
    }
    int ret = 0;
    while ((dir_ptr = readdir(pDir)) != 0){
        string tmp(dir_ptr->d_name);
        int idx = tmp.find_last_of('.');
        string suffix = tmp.substr(idx + 1, tmp.size() - idx - 1);
        if ("jpg" != suffix) continue;
        ++ ret;
    }
    closedir(pDir);
    return ret;
}

void OnlineDataLoader::threadRun() {
    DIR *pDir;
    struct dirent *dir_ptr;
    if (!(pDir = opendir(dataset_dir.c_str()))){
        throw runtime_error("dataset_dir not exists! dir : [" + dataset_dir + "] failed !" );
    }
    vector<string> files;
    while ((dir_ptr = readdir(pDir)) != 0){
        string tmp(dir_ptr->d_name);
        int idx = tmp.find_last_of('.');
        string suffix = tmp.substr(idx + 1, tmp.size() - idx - 1);
        if ("jpg" != suffix) continue;
        files.push_back(dataset_dir + "/" + tmp);
    }
    closedir(pDir);
    int total = files.size();

    int count = 0;
    // cv::Mat fake_image(224,224,0);
    shared_ptr<Mat> fake_image_ptr = make_shared<cv::Mat>(224,224,0);
    for (string& file_path : files){
        // 检查是否还在运行
        if (blockqueue_interface->isDisabled()){
            break;
        }
        auto image_ptr = make_shared<Mat>(cv::imread(file_path));
        if (image_ptr->empty()) continue;

        blockqueue_interface->put(image_ptr);
        ++ count;
        if (count == buffer_read_size) blockqueue_interface->customerEndWaiting();
    }

    blockqueue_interface->producerCallDone();

    m_runningThread.detach();
}

void OnlineDataLoader::start() {
    blockqueue_interface->enable();
    m_runningThread = std::thread(&OnlineDataLoader::threadRun, this);
}

void OnlineDataLoader::stop() {
    blockqueue_interface->disable();
    blockqueue_interface->clearQueue();
}

bool OnlineDataLoader::isRunning() {
    return m_runningThread.joinable();
}