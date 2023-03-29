#ifndef __GPU_STATUS_H
#define __GPU_STATUS_H

#include <array>
#include <iostream>
#include <memory>
#include <pthread.h>
#include "gpu_status.hpp"

using namespace std;

#define STATUS_BUFFER_SIZE 2048

class JetsonStatus : public IGpuStatus{
public:
    JetsonStatus(bool noPrint = true);
    int getRealTimePower() override;
    ~JetsonStatus();
    
private:
    const string tegrastats_path = "./tegrastats";
    array<char,STATUS_BUFFER_SIZE> status_buffer;
    shared_ptr<FILE> pipe;
    pthread_t thread;
    int record_power = 0;
    bool noPrint;


    bool file_exists(const std::string & name);    
    bool openPipe();
    bool initThread();
    static void* threadEntry(void* arg);
    void run();
};

#endif
