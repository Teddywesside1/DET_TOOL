#include "common.hpp"
using namespace std;

void Delay(int time)
{ 
    clock_t now = clock(); 

    while( (double)(clock() - now) / CLOCKS_PER_SEC * 1000 < time ); 
} 



string getSuffix(const string& str){
    int n = str.size();
    int idx = str.find_last_of('.');
    return idx == string::npos ? "" : str.substr(idx + 1, str.size() - idx - 1);
}