#include "temp.hpp"
#include "common.hpp"

TempModelInstance::TempModelInstance(const int delay, const int batch_size) : delay(delay), batch_size(batch_size){

}




void TempModelInstance::inferenceRounds(vector<shared_ptr<float>>& data_ptr) {
    Delay(delay);
    
}