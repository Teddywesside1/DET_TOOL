#include "hm800_status.h"

using namespace std;


HM800Status::HM800Status(bool noPrint) : noPrint(noPrint) {
    cout << "successfully create HM800Status Monitor !" << endl;
}



int HM800Status::getRealTimePower(){
    return record_power + (rand() % 4000) - 2000;
}


