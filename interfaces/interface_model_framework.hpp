/**
 * @file model_framework.hpp
 * @author binfeng.zhang
 * @brief interface of ModelFramework which provide functions to
 *          get some key information of the implemented framework
 * @version 0.1
 * @date 2023-07-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <memory>
#include <vector>

namespace ModelFramework{

class IModelFramework{
public:
    
    /**
     * @brief 
     * 
     * @param batch_size 
     */
    virtual void framework_forward(const int batch_size) = 0;

    /**
     * @brief Get the buffer object
     * 
     * @return std::vector<void*>& 
     */
    virtual std::vector<void*>& get_buffer() = 0;

    virtual ~IModelFramework() {}
};

} // ModelFramework