#pragma once

#include <vector>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <mutex>

#include <glog/logging.h>
#include <glog/log_severity.h>



template <typename _Type>
class BlockQueue {
public:
    // typedef _Type _DataType;
    typedef _Type&              _reference;
    typedef const _Type&        _const_reference;
    typedef _Type               _value_type;

    BlockQueue(const int maxQueueSize);

    _value_type pop();

    void push(const _value_type& value);

    void push(const _value_type&& value);

    int size() ;

    bool empty() ;

private:
    std::queue<_Type> _q;
    const int _maxQueueSize;

    std::mutex _queue_lock;
    std::condition_variable _consumer_cv;
    std::condition_variable _producer_cv;
};




template <typename _Type>
BlockQueue<_Type>::BlockQueue(const int maxQueueSize)
                            : _maxQueueSize(maxQueueSize){

}

template <typename _Type>
typename BlockQueue<_Type>::_value_type 
BlockQueue<_Type>::pop(){
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    while (_q.empty()){
        _consumer_cv.wait(u_lck);
    }
    _value_type value = _q.front();
    _q.pop();
    _producer_cv.notify_one();

    return std::move(value);
}

template <typename _Type>
void 
BlockQueue<_Type>::push(const _value_type& value){
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    while (_q.size() >= _maxQueueSize){
        _producer_cv.wait(u_lck);
    }
    _q.push(value);
    _consumer_cv.notify_one();
}

template <typename _Type>
void 
BlockQueue<_Type>::push(const _value_type&& value){
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    while (_q.size() >= _maxQueueSize){
        _producer_cv.wait(u_lck);
    }
    _q.push(value);
    _consumer_cv.notify_one();
}



template <typename _Type>
int 
BlockQueue<_Type>::size() {
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    return _q.size();
}



template <typename _Type>
bool 
BlockQueue<_Type>::empty() {
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    return _q.empty();
}