#ifndef __BLOCK_QUEUE_H
#define __BLOCK_QUEUE_H

#include <opencv2/opencv.hpp>

#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
// #include <cuda_runtime_api.h>
#include <iostream>

using namespace std;
using namespace cv;

class BlockQueueInterface{
public:
    virtual void put(shared_ptr<Mat> ptr) = 0;
    virtual shared_ptr<Mat> take() = 0;
    virtual void disable() = 0;
    virtual void enable() = 0;
    virtual void clearQueue() = 0;
    virtual bool isDisabled() = 0;
    virtual void producerCallDone() = 0;
    virtual void customerEndWaiting() = 0;
};


class BlockQueue : public BlockQueueInterface{
private:
    queue<shared_ptr<Mat>> q;
    mutex lock;
    condition_variable cv_cons;
    condition_variable cv_prod;
    atomic<bool> stopped;
    atomic<bool> producer_done_job;
    atomic<bool> customer_wait;

    const int capacity;

public:
    BlockQueue(const int capacity = 2000);
    ~BlockQueue();

    void clearQueue(){
        unique_lock<mutex> lck(lock);
        while (!q.empty()){
            q.pop();
        }
    }

    shared_ptr<Mat> take() override;
    void put(shared_ptr<Mat> ptr) override;
    
    void enable() override {
        stopped.store(false);
        producer_done_job.store(false);
        customer_wait.store(true);
    }

    void customerEndWaiting(){
        customer_wait.store(false);
        cv_cons.notify_all();
    }

    bool isDisabled(){
        return stopped.load();
    }

    void disable(){
        stopped.store(true);
        cv_cons.notify_all();
        cv_prod.notify_all();
    }

    void producerCallDone(){
        producer_done_job.store(true);
        cv_cons.notify_all();
        cv_prod.notify_all();
    }

    bool clientCheckDone(){
        return producer_done_job.load();
    }

    bool isEmpty(){
        // unique_lock<mutex> lck(lock);
        return q.empty();
    }

    bool isFull(){
        // unique_lock<mutex> lck(lock);
        return q.size() == capacity;
    }
};


#endif
