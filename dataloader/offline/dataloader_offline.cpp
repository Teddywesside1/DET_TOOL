#include "dataloader_offline.hpp"
#include <dirent.h>
#include "common.hpp"


OfflineDataLoader::OfflineDataLoader(const string dataset_dir, const int frame_total, const int round_total) 
                        : dataset_dir(dataset_dir), frame_total(frame_total), round_total(round_total){

}


vector<shared_ptr<float>> OfflineDataLoader::getOneRound(const int batch_count_one_round, const int batch_size, const int single_input_size, 
                                                        std::function<void(shared_ptr<float>, cv::Mat&, const int)> place_method){
    static vector<vector<shared_ptr<float>>> roundsPtrArr;
    static int count = 0;
    if (roundsPtrArr.size() == 0){
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

        int image_idx = 0;
        const int image_total = files.size();
        for (int r = 0 ; r < round_total ; ++ r){
            vector<shared_ptr<float>> roundsPtr;
            for (int i = 0 ; i < batch_count_one_round ; ++ i){
                shared_ptr<float> data_ptr{new float[single_input_size * batch_size], [](float* ptr){delete ptr;}};
                for (int j = 0 ; j < batch_size ; ++ j){
                    auto image = cv::imread(files[image_idx++ % image_total]);
                    if (image.empty()){
                        continue;
                    }
                    place_method(data_ptr,image,j);
                }
                roundsPtr.push_back(data_ptr);
            }
            roundsPtrArr.push_back(move(roundsPtr));
        }
    }

    if (running_flag.load() == false || count * batch_count_one_round > frame_total){
        running_flag.store(false);
        count = 0;
        return vector<shared_ptr<float>>();
    }

    return roundsPtrArr[count ++ % round_total];

}





int OfflineDataLoader::getDatasetTotal(){
    return frame_total;
}


void OfflineDataLoader::start() {
    running_flag.store(true);
}

void OfflineDataLoader::stop() {
    running_flag.store(false);
}

bool OfflineDataLoader::isRunning() {
    return running_flag.load();
}