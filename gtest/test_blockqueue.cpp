#include <gtest/gtest.h>
#include "util_data_structure.hpp"
#include <thread>


TEST(data_structure_test, block_queue)
{
    const int max_queue_size = 5;
    BlockQueue<int> block_queue(max_queue_size);
    int idx = 0;

    auto producer_thread = [&](){
        constexpr int total_to_push = 2 * max_queue_size;
        for (int i = 0 ; i < total_to_push ; ++ i){
            block_queue.push(idx++);
        }
    };

    std::thread producer_thread_handler(producer_thread);
    producer_thread_handler.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_EQ(block_queue.size(), max_queue_size);

    int cur = block_queue.pop();
    ASSERT_EQ(cur, 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(block_queue.size(), max_queue_size);

    for (int i = 0 ; i < max_queue_size ; ++ i){
        int ret = block_queue.pop();
        ASSERT_EQ(ret, i + 1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(block_queue.size(), max_queue_size - 1);



}