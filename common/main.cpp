#include "logger.hpp"
#include <iostream>

int main(){
    int i = 1;
    logger << LogLevel::INFO << "abc " << i << LOG_LINE_END;

}