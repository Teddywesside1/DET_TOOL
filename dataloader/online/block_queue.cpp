#include "block_queue.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


BlockQueue::BlockQueue(const int capacity) : capacity(capacity){
    stopped.store(true);
    producer_done_job.store(false);
}

BlockQueue::~BlockQueue(){
    clearQueue();
}

void BlockQueue::put(shared_ptr<Mat> ptr){
    unique_lock<mutex> lck(lock);
    while (isFull() && !isDisabled()){
        cv_prod.wait(lck);
    }
    if (isDisabled()) return;

    q.push(ptr);
    cout << "-- block_queue size : " << q.size() << endl;
    cv_cons.notify_one();
}


shared_ptr<Mat> BlockQueue::take(){
    unique_lock<mutex> lck(lock);
    while ( (customer_wait.load() || isEmpty()) && (!isDisabled() && !clientCheckDone()) ){
        cv_cons.wait(lck);
    }
    
    if (isDisabled() || (isEmpty() && clientCheckDone())) return NULL;

    shared_ptr<Mat> ret = q.front();
    q.pop();
    cv_prod.notify_one();
    return ret;
}


