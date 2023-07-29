#pragma once

#include <vector>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <optional>

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

    std::optional<_value_type> pop();

    void push(const _value_type& value);

    void push(const _value_type&& value);

    int size() ;

    bool empty() ;

    void callStop();

private:
    std::queue<_Type> _q;
    const int _maxQueueSize;

    std::mutex _queue_lock;
    std::condition_variable _consumer_cv;
    std::condition_variable _producer_cv;

    std::atomic<bool> _stop_flag {false};
};




template <typename _Type>
BlockQueue<_Type>::BlockQueue(const int maxQueueSize)
                            : _maxQueueSize(maxQueueSize){

}

template <typename _Type>
std::optional<typename BlockQueue<_Type>::_value_type>
BlockQueue<_Type>::pop(){
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    while (_q.empty() && !_stop_flag.load()){
        _consumer_cv.wait(u_lck);
    }
    if (_stop_flag.load() && _q.empty()){
        return std::nullopt;
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
    while (_q.size() >= _maxQueueSize && !_stop_flag.load()){
        _producer_cv.wait(u_lck);
    }
    if (_stop_flag.load()){
        return;
    }
    _q.push(value);
    _consumer_cv.notify_one();
}

template <typename _Type>
void 
BlockQueue<_Type>::push(const _value_type&& value){
    std::unique_lock<std::mutex> u_lck(_queue_lock);
    while (_q.size() >= _maxQueueSize && !_stop_flag.load()){
        _producer_cv.wait(u_lck);
    }
    if (_stop_flag.load()){
        return;
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


template <typename _Type>
void 
BlockQueue<_Type>::callStop() {
    _stop_flag.store(true);
    _producer_cv.notify_all();
    _consumer_cv.notify_all();
}