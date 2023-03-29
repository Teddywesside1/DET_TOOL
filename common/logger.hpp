#pragma once

#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

#define LOG_LINE_END '\n'


enum LogLevel{
    INFO,
    WARN,
    ERRO
};

class Logger {
public:
    Logger(const string log_file = "log.txt", const bool printConsole = true, const bool writeFile = false);

    template<typename T>
    Logger& operator<<(const T& s);

    Logger& operator<<(const LogLevel& l);

private:
    template<typename T>
    void print(const T& s);

    template<typename T>
    void write(const T& s);

private:
    const bool printConsole;
    const bool writeFile;

    ofstream m_fileStream;

};

extern Logger logger;


template<typename T>
Logger& Logger::operator<<(const T& s){
    print(s);
    write(s);
    return *this;
}


template<typename T>
void Logger::print(const T& s){
    if (!printConsole) return ;

    cout << s << " ";
}

template<typename T>
void Logger::write(const T& s){
    if (!writeFile) return ;

    m_fileStream << s << " ";
    m_fileStream.flush();
}
