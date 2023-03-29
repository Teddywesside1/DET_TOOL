#include "jetson_status.h"
#include <iostream>
#include <wordexp.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <unistd.h>

using namespace std;


JetsonStatus::JetsonStatus(bool noPrint) : noPrint(noPrint) {
    if (!file_exists(tegrastats_path)) {
        throw runtime_error("-- [JetsonStatus] tegrastats path not exists !");
    }    
    string command = tegrastats_path + " --interval 200" ;
    pipe.reset(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw runtime_error("-- [JetsonStatus] openPipe failed !");

    if (!initThread()) {
        throw runtime_error("-- [JetsonStatus] init thread failed !");
    }    

    cout << "successfully create JetsonStatus Monitor !" << endl;
}

JetsonStatus::~JetsonStatus(){

}

bool JetsonStatus::openPipe(){
    
}

bool JetsonStatus::initThread(){
    if (pthread_create(&thread,NULL,threadEntry,this) != 0){
        return false;
    }
    if (pthread_detach(thread)){
        return false;
    }
    return true;
}

void* JetsonStatus::threadEntry(void* arg){
    JetsonStatus* ins_ptr = (JetsonStatus*) arg;
    ins_ptr->run();
    return ins_ptr;
}

void JetsonStatus::run(){
    fd_set read_set;
    int fd = fileno(pipe.get());
    FD_ZERO(&read_set);
    FD_SET(fd,&read_set);

    while(1){
        int ret = select(1+fd,&read_set,NULL,NULL,NULL);
        if (ret > 0 && FD_ISSET(fd,&read_set)){
            fgets(status_buffer.data(), STATUS_BUFFER_SIZE, pipe.get());
            string status_output = status_buffer.data();
            int idx = status_output.find("VDD_GPU_SOC");
            if (idx == string::npos) continue;
            idx += 12;
            int power = 0;
            while (isdigit(status_output[idx])){
                power = power * 10 + status_output[idx] - '0';
                ++ idx;
            }
            
            record_power = power;
            if (!noPrint)
                cout << power << endl;
            
        }else{
            FD_SET(fd,&read_set);
        }
    }
}


int JetsonStatus::getRealTimePower(){
    return record_power;
}

bool JetsonStatus::file_exists(const std::string & name) {
  struct stat buffer;
  std::string full_name;
  wordexp_t expanded_name;

  wordexp(name.c_str(), &expanded_name, 0);
  full_name = expanded_name.we_wordv[0];
  wordfree(&expanded_name);

  return (stat(full_name.c_str(), &buffer) == 0);
}

