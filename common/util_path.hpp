#pragma once

#include <string>
#include <glog/logging.h>
#include <glog/log_severity.h>

inline std::string 
get_suffix(const std::string & s){
    int n = s.size();
    int idx = s.find_last_of('.');
    return idx == std::string::npos ? "" : s.substr(idx + 1, s.size() - idx - 1);
}

int get_files_number_in_dir(const std::string & dir);