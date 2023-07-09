#pragma once

#include <string>

inline std::string 
get_suffix(const std::string & s){
    int n = s.size();
    int idx = s.find_last_of('.');
    return idx == std::string::npos ? "" : s.substr(idx + 1, s.size() - idx - 1);
}