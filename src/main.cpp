#include "CMakeConfig.h"

#include "crow_all.h"
#include "models_manager.hpp"
#include "container.hpp"
#include "yolov3.hpp"
#include "resnet50.hpp"
#include "logger.hpp"
#if (WITHOUT_IMAGE_READING)
    #include "dataloader_fake.hpp"
#else
    #include "dataloader_offline.hpp"
    #include "dataloader_online.hpp"
#endif

#if (SERVER_PLATFORM == 0)
    #include "jetson_status.h"
    #include "TRT_model_config.hpp"
#else
    #include "temp.hpp"
    #include "hm800_model.hpp"
    #include "hm800_status.h"
#endif


using namespace std;

int main(){

    const string dataset_dir = DATASET_DIR;
    const int yolo_frame_total = INT_MAX, res50_frame_total = INT_MAX;
    const string yolo_alias = "Yolo-V3", res50_alias = "Resnet-50";
    const int input_channel = 3;

#if (SERVER_PLATFORM == 0)
    const string yolo_model_path = "yolov3-10-without_yolo.onnx_int8.engine";
    const string yolo_input_blob_name = "input_1";
    const int yolo_batch_size = 4, yolo_round = 5, yolo_input_height = 416, yolo_input_width = 416;
    const string res50_model_path = "resnet50-v1-12.onnx_int8.engine";
    const string res50_input_blob_name = "data";
    const int res50_batch_size = 32, res50_round = 20, res50_input_height = 224, res50_input_width = 224;
#else
    const string yolo_model_path = "libyolov3";
    const string yolo_input_blob_name = "data";
    const int yolo_batch_size = 4, yolo_round = 100, yolo_input_height = 416, yolo_input_width = 416;
    const string res50_model_path = "resnet50";
    const string res50_input_blob_name = "Input.1";
    const int res50_batch_size = 32, res50_round = 200, res50_input_height = 224, res50_input_width = 224;
#endif
    


#if (WITHOUT_IMAGE_READING)
    shared_ptr<DataLoaderFactoryBase> yolo_dataloader_factory = make_shared<FakeDataLoaderFactory>
        (dataset_dir,yolo_frame_total);
    shared_ptr<DataLoaderFactoryBase> res50_dataloader_factory = make_shared<FakeDataLoaderFactory>
        (dataset_dir,res50_frame_total);
#else
    shared_ptr<DataLoaderFactoryBase> yolo_dataloader_factory = make_shared<OfflineDataLoaderFactory>
        (dataset_dir,yolo_frame_total,1);
    shared_ptr<DataLoaderFactoryBase> res50_dataloader_factory = make_shared<OfflineDataLoaderFactory>
        (dataset_dir,res50_frame_total,1);    
    // shared_ptr<DataLoaderFactoryBase> dataloader_factory = make_shared<OnlineDataLoaderFactory>
        // (dataset_dir);
#endif


#if (SERVER_PLATFORM == 0)
    shared_ptr<ITRTModelConfig> yolov3_config;
    yolov3_config = make_shared<TRTModelConfig_INT8>(yolov3_config);
    yolov3_config = make_shared<TRTModelConfig_NoDataTransfers>(yolov3_config);
    yolov3_config = make_shared<TRTModelConfig_DynamicBatch>(yolov3_config);
    IModelInstance* yolov3_model_instance = new TRTModelInstance(yolo_model_path, yolo_input_blob_name, yolo_batch_size, yolo_input_height, yolo_input_width, input_channel, yolov3_config);
#else
    // IModelInstance* yolov3_model_instance = new TempModelInstance(300,4);
    IModelInstance* yolov3_model_instance = new HM800ModelInstance(yolo_model_path,yolo_input_blob_name,yolo_batch_size,yolo_input_height,yolo_input_width,input_channel,yolo_round,1395);

#endif
    shared_ptr<ModelRuntimeFactoryBase> yolov3_model_runtime_factory = make_shared<Yolov3Factory>
        (yolo_batch_size, yolov3_model_instance);
    shared_ptr<Container> yolov3_container = make_shared<Container>(yolo_alias, yolo_batch_size, yolo_round, yolov3_model_runtime_factory, yolo_dataloader_factory);


#if (SERVER_PLATFORM == 0)
    shared_ptr<ITRTModelConfig> res50_config;
    res50_config = make_shared<TRTModelConfig_FP16>(res50_config);
    res50_config = make_shared<TRTModelConfig_NoDataTransfers>(res50_config);
    res50_config = make_shared<TRTModelConfig_DynamicBatch>(res50_config);
    IModelInstance* res50_model_instance = new TRTModelInstance(res50_model_path, res50_input_blob_name, res50_batch_size, res50_input_height, res50_input_width, input_channel, res50_config);
#else
    // IModelInstance* res50_model_instance = new TempModelInstance(3,32);
    IModelInstance* res50_model_instance = new HM800ModelInstance(res50_model_path,res50_input_blob_name,res50_batch_size,res50_input_height,res50_input_width,input_channel,res50_round,10066);
#endif
    shared_ptr<ModelRuntimeFactoryBase> res50_model_runtime_factory = make_shared<ResNet50Factory>
        (res50_batch_size, res50_model_instance);
    shared_ptr<Container> res50_container = make_shared<Container>(res50_alias, res50_batch_size, res50_round, res50_model_runtime_factory, res50_dataloader_factory);
    
    Manager manager;
    manager.createModel(res50_container->getName(),res50_container);
    manager.createModel(yolov3_container->getName(),yolov3_container);


    
#if (SERVER_PLATFORM == 0)
    shared_ptr<IGpuStatus> powerMonitor = make_shared<JetsonStatus>();
#else
    shared_ptr<IGpuStatus> powerMonitor = make_shared<HM800Status>();
#endif

    crow::SimpleApp app;
    app.loglevel(crow::LogLevel::Warning);
    CROW_ROUTE(app, POST_URL).methods("POST"_method)
        ([&](const crow::request& req){
            auto x = crow::json::load(req.body);
            if (!x)
                return crow::response(400);
            
            string command = x["command"].s();
            crow::json::wvalue resp;
            logger << LogLevel::INFO << "request from :" << req.remoteIpAddress << "command :" << command << LOG_LINE_END;
            if ("getStatus" == command){

                int power = powerMonitor->getRealTimePower();

                // 构建json响应报文
                resp["res_command"] = "getStatus";
                resp["response"] = true;
                resp["cur_status"] = manager.isRunning() ? "running" : "stop";
                resp["power"] = power;

                vector<Manager::ModelInfo> __models_info;
                manager.getAllModelsInfo(__models_info);
                vector<crow::json::wvalue> models_info;
                for (auto &info : __models_info){
                    crow::json::wvalue model_info;
                    model_info["model"] = info.model_name;
                    model_info["cur_status"] = info.running ? "running" : "stop";
                    crow::json::wvalue progress;
                    progress["total"] = info.total;
                    progress["cur"] = info.cur;
                    progress["FPS"] = info.FPS;
                    progress["batch_size"] = info.batch_size;
                    model_info["progress"] = move(progress);
                    models_info.push_back(move(model_info));
                }
                resp["models_progress"] = move(models_info);

            }
            else if ("start" == command){
                string model_name = x["model"].s();
                bool flag = manager.startModel(model_name);  
                string cur_status = manager.isRunning() ? "running" : "stop";              
                resp["res_command"] = "start";
                resp["response"] = flag;
                resp["cur_status"] = move(cur_status);
            }
            else if ("stop" == command){
                string model_name = x["model"].s();
                bool flag = manager.stopModel(model_name);
                string cur_status = manager.isRunning() ? "running" : "stop";
                resp["res_command"] = "stop";
                resp["response"] = flag;
                resp["cur_status"] = move(cur_status);
            }
            else if ("resetStatus" == command){
                string model_name = x["model"].s();
                bool flag = manager.resetProgress(model_name);
                string cur_status = manager.isRunning() ? "running" : "stop";
                resp["res_command"] = "resetStatus";
                resp["response"] = flag;
                resp["cur_status"] = move(cur_status);
            }

            return crow::response{resp};
        });

    app.port(LISTEN_PORT).multithreaded().run();
}
