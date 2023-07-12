#include "util_path.hpp"
#include <dirent.h>
#include <exception>

// std::string 
// get_suffix(const std::string & s){
//     int n = s.size();
//     int idx = s.find_last_of('.');
//     return idx == std::string::npos ? "" : s.substr(idx + 1, s.size() - idx - 1);
// }




int 
get_files_number_in_dir(const std::string & dir){
    DIR *pDir;
    struct dirent *dir_ptr;
    if (!(pDir = opendir(dir.c_str()))){
        LOG(ERROR) << "failed to open dir `" << dir << "`, please check if the path is a valid dir path!";
        throw std::runtime_error("failed to open dir");
    }
    int ret = 0;
    while ((dir_ptr = readdir(pDir)) != 0){
        ++ ret;
    }
    if (!closedir(pDir)){
        LOG(ERROR) << "failed to close dir `" << dir << "` !";
        throw std::runtime_error("failed to close dir");
    }
    return ret;
}