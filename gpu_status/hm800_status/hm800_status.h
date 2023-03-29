#ifndef __GPU_STATUS_H
#define __GPU_STATUS_H

#include <array>
#include <iostream>
#include <memory>
#include "gpu_status.hpp"

using namespace std;

#define STATUS_BUFFER_SIZE 2048

class HM800Status : public IGpuStatus{
public:
    HM800Status(bool noPrint = true);
    int getRealTimePower() override;
    
private:
    bool noPrint;
    int record_power = 20000;
};

#endif
