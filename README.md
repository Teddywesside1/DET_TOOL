# server_multi-platform 
## 环境准备
### 必要
1.  Boost
-   由Apollo下houmoai.sh脚本创建的容器需要执行
    ```console
        ~$ cp -r /usr/include/boost.old /usr/include/boost
    ```
-   容器内安装boost
    ```console
        ~$ apt-get update
        ~$ apt-get install libboost-all-dev
    ```
    阿里源可以找到libboost-all-dev包, 清华源不行

### Orin平台
1.  CUDA、TensorRT
-   更改根目录下CMakeLists.txt中的 `SERVER_PLATFORM` 定义, 来选择编译的平台： 0 - ORin, 1 - Others

2.  Jetson Tegrastats
-   将Orin宿主机下的 `tegrastats` 二进制文件映射或拷贝到容器内
    ```console
        ~$ cp /usr/bin/tegrastats ${REPO}/build/tegrastats
    ```

### HM800平台
1.  工具链
-   `http://10.10.1.53:8082/ui/repos/tree/General/toolchain%2Ftmp%2Fplatform.tar.gz`
    ```console
        ~$ source ${REPO}/toolchain/env.sh
        ~$ export HDPL_PLATFORM=ISIM / export HDPL_PLATFORM=ASIC
    ```
    定义环境变量 `HDPL_PLATFORM` , `ISIM` - 运行在仿真环境, `ASIC` 运行在硬件环境

2.  准备模型
-   在 `${REPO}/toolchain/itvm/benchmarks/cpp/` 下的 `multi_batch_fusedop` 和 `yolov3` 文件夹下按照脚本生成模型
-   复制后缀为 `.so` 的文件到 `${server}/build/` 下


### 非必要
1.  OpenCV
-   推理使用随机输入数据时不需要OpenCV参与编译, 更改根目录下CMakeLists.txt中的 `WITHOUT_IMAGE_READING` 定义, 来取消对OpenCV的依赖


## 构建
    ```console
        ~$ mkdir build
        ~$ cd build
        ~$ cmake ..
        ~$ make
    ```

## HTTP接口
### 请求
1.  请求行: `POST 10.64.32.14:11188/orin_request` / `POST 10.64.34.181:11190/hm800_request`
2.  请求体：
    ```json
        {
            "command"   : "start" / "stop" / "getStatus",
            "model"     : "yolov3" / "res50"
        }
    ```

### 响应
1.  响应体：
    ```json
        {
            "res_command"   : "start" / "stop" / "getStatus" / "resetStatus",   // 表明响应的命令
            "response"      : true / false,                                     // 命令是否执行成功
            "cur_status"    : "running" / "stop",                               // 当前运行状态
            "model"         : "Yolo-V3" / "Resnet-50" / ... ,                   // 当前运行的模型类别
            "power"         : (int) ,                                           // 当前gpu实时功率
            "progress"      : {
                                "total" : (int),                // 测试集总数
                                "cur"   : (int),                // 当前推理进度
                                "FPS"   : (int),                // 平均FPS
                                "batch" : (int)                 // batch_size
                            }
        }
    ```


## 部署新的模型(Orin平台)
1.  模型生成配置
    ```cpp
        shared_ptr<ITRTModelConfig> config;
        config = make_shared<TRTModelConfig_INT8>(config);              // 转换为INT8精度
        config = make_shared<TRTModelConfig_NoDataTransfers>(config);   // 禁止Host2Device和Device2Host数据拷贝过程
        config = make_shared<TRTModelConfig_DynamicBatch>(config);      // onnx文件使用动态batch_size
    ```
    动态batch_size范围默认1~32, 优化的batch_size默认是生成时输入的batch_size

2.  生成TensorRT模型实例
    ```cpp
        IModelInstance* trt_model_instance = new TRTModelInstance(model_path, input_blob_name, batch_size, rounds, input_height, input_width, input_channel, config);
    ```
    若 `model_path` 后缀为 `.onnx` 会重新生成一个后缀为 `.engine` 的文件
    若 `model_path` 后缀为 `.engine` 会直接读取

3.  生成model_runtime实例
    ```cpp
        IModelRuntime* model = new Yolov3(batch_size, trt_model_instance);  
    ```
    编写新的模型class继承自IModelRuntime, 引用trt_model_instance实例

4.  生成dataloader实例
    ```cpp
        IDataLoader* dataloader = new OfflineDataLoaderFactory(dataset_dir,frame_total,round_total);
        IDataLoader* dataloader = new OnlineDataLoaderFactory(dataset_dir);     // 还有个bug
    ```
    OfflineDataLoader在读取一部分图片后循环使用
    OnlineDataLoader利用阻塞队列实现

5.  生成dataloader和model_runtime的container

6.  向Manager注册模型
    ```cpp
        manager.createModel(container->getName(),container);
    ```



