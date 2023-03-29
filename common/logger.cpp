#include "logger.hpp"

Logger logger;

Logger::Logger(const string log_file, const bool printConsole, const bool writeFile) : printConsole(printConsole), writeFile(writeFile){
    m_fileStream.open(log_file, ios_base::app);
    if (writeFile && !m_fileStream.is_open()) {
        throw runtime_error("can not open log file !");
    }
}

Logger& Logger::operator<<(const LogLevel& l){

    time_t now = std::time(nullptr);
    tm time_info = *std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S]", &time_info);
    string s(buffer);
    print(s);
    write(s);
    switch (l){
        case INFO:
            print("[INFO]");
            write("[INFO]");
            break;
        case WARN:
            print("[WARN]");
            write("[WARN]");
            break;
        case ERRO:
            print("[ERRO]");
            write("[ERRO]");
            break;
        default:
        break;
    }
    
    return *this;
}