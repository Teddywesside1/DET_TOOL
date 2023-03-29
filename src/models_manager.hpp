#pragma once

#include <unordered_map>
#include <vector>
#include "container.hpp"

using namespace std;

class Manager{
public:
    struct ModelInfo{
        const string model_name;
        const bool running;
        const int total;
        const int cur;
        const int FPS;
        const int batch_size;
        ModelInfo(const string model_name, const bool running, const int total, const int cur, const int FPS, const int batch_size) : 
                    model_name(model_name), running(running), total(total), cur(cur), FPS(FPS), batch_size(batch_size) {}
    };

    Manager();

    bool startModel(const string& model_name);

    bool stopModel();

    bool stopModel(const string model_name);

    bool getProgress(const string model_name, int &total, int &cur, int &FPS, int &batch_size);

    bool resetProgress(const string model_name);

    bool createModel(const string& model_name, shared_ptr<Container> container_ptr);

    void getAllModelsInfo(vector<ModelInfo> &models_progress);

    bool isRunning();

    const string getRunningModelName();
private:

    unordered_map<string,shared_ptr<Container>> table;

    shared_ptr<Container> running_model = NULL;
};
