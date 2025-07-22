#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <regex>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>

#define clock_                           // 用于性能测试计时
#define min(a, b) (((a) < (b)) ? (a) : (b))

// 模型初始化参数
typedef struct PARAMS{
    std::string modelPath;                 // 模型文件路径
    std::vector<int> imgSize = {640, 640}; // 输入图像尺寸，默认为 640x640
    float rectConfidenceThreshold = 0.6;   // 目标检测置信度阈值，低于此值的框会被过滤
    float iouThreshold = 0.5;              // IoU阈值，用于非极大值抑制(NMS)
    int keyPointsNum = 2;                  // 关键点数量，用于姿态检测模型
    int logSeverityLevel = 3;              // 日志级别
    int intraOpNumThreads = 1;             // 内部并行操作的线程数
    std::vector<std::string> classes{
        "Raw_Banana","Raw_Mango","Ripe_Banana","Ripe_Mango"
    };    // 类别
} PARAMS;

// 推理结果结构体，用于存储检测、分类或姿态检测的结果
typedef struct RESULT{
    int classId;                        // 类别ID
    float confidence;                   // 置信度分数
    cv::Rect box;                       // 边界框（检测结果的位置）
} RESULT;

class MODEL{
public:
    MODEL();
    ~MODEL();
public:
    // 创建ONNX会话，并加载模型
    void CreateSession(PARAMS &iParams);
    // 执行推理，处理输入图像并返回结果
    void RunSession(cv::Mat &iImg, std::vector<RESULT> &oResult);
    // 预热，用于首次运行模型时的性能优化
    void WarmUpSession();
    // 后处理
    void TensorProcess(clock_t &starttime_1, cv::Mat &iImg, float* &blob, std::vector<int64_t> &inputNodeDims,
                        std::vector<RESULT> &oResult);
    // 预处理
    void PreProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg);
    // 类别名称
    std::vector<std::string> classes{};

private:
    Ort::Env env;                              // ONNX Runtime环境
    Ort::Session *session;                     // ONNX会话，用于执行模型推理
    Ort::RunOptions options;                   // 运行选项
    std::vector<const char *> inputNodeNames;  // 输入节点名称
    std::vector<const char *> outputNodeNames; // 输出节点名称

    std::vector<int> imgSize;      // 图像尺寸
    float rectConfidenceThreshold; // 置信度阈值
    float iouThreshold;            // IoU阈值
    float resizeScales;            // 缩放比例
};

MODEL::MODEL(){}
MODEL::~MODEL(){delete session;}

/**
 * 将OpenCV图像转换为模型输入所需的Blob格式
 * @param iImg 输入图像
 * @param iBlob 输出Blob
 */

void BlobFromImage(cv::Mat &iImg, float* &iBlob){
    int channels = iImg.channels(); // 通道数
    int imgHeight = iImg.rows;      // 高度
    int imgWidth = iImg.cols;       // 宽度

    // 逐像素填充Blob，同时进行归一化
    for (int c = 0; c < channels; c++)
        for (int h = 0; h < imgHeight; h++)
            for (int w = 0; w < imgWidth; w++)
                // 将图像数据转换为浮点数并归一化到[0,1]范围
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] =
                 float((iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
}

/**
 * 图像预处理函数：调整大小、转换颜色空间等
 * @param iImg 输入原始图像
 * @param iImgSize 目标图像尺寸
 * @param oImg 输出预处理后的图像
 */
void MODEL::PreProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg){
    if (iImg.channels() == 3)
        cv::cvtColor(iImg, oImg, cv::COLOR_BGR2RGB);
    else if (iImg.channels() == 1)
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);

    // LetterBox处理
    if (iImg.cols >= iImg.rows){
        resizeScales = iImg.cols / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
    }
    else{
        resizeScales = iImg.rows / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
    }
    // 创建目标尺寸的黑色图像，将调整后的图像放入其中（添加黑边）
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;
}

/**
 * 创建ONNX会话并加载模型
 * @param iParams 初始化参数
 */
void MODEL::CreateSession(PARAMS& iParams) {
    // 保存参数到类成员变量
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    classes = iParams.classes;
    const char* modelPath = iParams.modelPath.c_str();
    // 创建ONNX Runtime环境
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "MODEL");
    Ort::SessionOptions sessionOption;

    // 设置图优化级别和线程数
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // 启用所有优化
    sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads); // 设置内部操作线程数
    sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel); // 设置日志级别

    // 创建ONNX会话，加载模型
    session = new Ort::Session(env, modelPath, sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 获取模型的输入节点名称
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    
    // 获取模型的输出节点名称
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++){
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    
    // 设置运行选项并预热会话
    options = Ort::RunOptions{ nullptr };
    WarmUpSession();
}

/**
 * 运行推理会话
 * @param iImg 输入图像
 * @param oResult 输出推理结果
 */
void MODEL::RunSession(cv::Mat& iImg, std::vector<RESULT>& oResult) {
#ifdef clock_
    clock_t starttime_1 = clock();  // 开始计时，用于性能测试
#endif

    cv::Mat processedImg;
    // 预处理输入图像
    PreProcess(iImg, imgSize, processedImg);

    float* blob = new float[processedImg.total() * 3];
    BlobFromImage(processedImg, blob);  // 将处理后的图像转换为Blob
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };  // 设置输入维度
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);  // 执行推理和后处理

}

/**
 * 处理张量数据并执行推理和后处理
 * @param starttime_1 开始时间，用于性能测量
 * @param iImg 原始输入图像
 * @param blob 预处理后的Blob数据
 * @param inputNodeDims 输入节点维度
 * @param oResult 输出结果
 */

void MODEL::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, float* & blob, std::vector<int64_t>& inputNodeDims,
    std::vector<RESULT>& oResult) {
    
    // 创建输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 
        3 * imgSize.at(0) * imgSize.at(1),inputNodeDims.data(), inputNodeDims.size());
        
#ifdef clock_
    clock_t starttime_2 = clock();  // 记录预处理完成时间
#endif // clock_

    // 执行模型推理
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
        
#ifdef clock_
    clock_t starttime_3 = clock();  // 记录推理完成时间
#endif // clock_

    // 获取输出张量信息
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<float>();
    
    // 释放Blob内存
    delete[] blob;
    
    int signalResultNum = outputNodeDims[1];  // 每个检测框的数据量
    int strideNum = outputNodeDims[2];        // 检测框的数量
    
    // 存储后处理数据的容器
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // 将输出数据转换为OpenCV矩阵以便处理
    cv::Mat rawData;

    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);

    // 转置矩阵
    rawData = rawData.t();
    float* data = (float*)rawData.data;

    // 解析每个检测框
    for (int i = 0; i < strideNum; ++i){
        float* classesScores = data + 4;  // 前4个是坐标
        cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        
        // 找出最高分数的类别
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        
        // 如果最高分数超过阈值，则保存这个检测框
        if (maxClassScore > rectConfidenceThreshold){
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            
            // 解析坐标（输出中心点坐标和宽高）
            float x = data[0];  // 中心x坐标
            float y = data[1];  // 中心y坐标
            float w = data[2];  // 宽度
            float h = data[3];  // 高度

            // 转换为左上角坐标和宽高格式，并应用缩放比例
            int left = int((x - 0.5 * w) * resizeScales);
            int top = int((y - 0.5 * h) * resizeScales);
            int width = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // 移动到下一个检测框数据
        data += signalResultNum;
    }
    
    // NMS消除重叠的框
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
    
    // 整理最终结果
    for (int i = 0; i < nmsResult.size(); ++i){
        int idx = nmsResult[i];
        RESULT result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        oResult.push_back(result);
    }

#ifdef clock_
    // 计算并输出性能指标
    clock_t starttime_4 = clock();
    double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
    std::cout << "[MODEL]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
#endif

}

/**
 * 预热会话函数，用于优化首次推理性能
 * 创建一个空白图像执行一次推理，以便ONNX Runtime完成初始化
 */
void MODEL::WarmUpSession() {
    clock_t starttime_1 = clock();
    
    // 创建一个空白图像
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    
    // 预处理图像
    PreProcess(iImg, imgSize, processedImg);
    
    float* blob = new float[iImg.total() * 3];
    BlobFromImage(processedImg, blob);
    std::vector<int64_t> input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        input_node_dims.data(), input_node_dims.size());// 创建输入张量
    auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
        outputNodeNames.size()); // 执行推理
        
    delete[] blob;
    clock_t starttime_4 = clock();
    double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
}

/**
 * 目标检测演示函数：读取images目录下的图片，运行检测模型，并显示结果
 * @param p MODEL对象指针
 */
void Detect(MODEL*& p) {
    std::string img_path = "../images/Raw_Mango_0_5537.jpg";
    cv::Mat img = cv::imread(img_path);
    std::vector<RESULT> res;
    // 获取检测结果
    p->RunSession(img, res);
    // 处理每个检测结果
    for (auto& re : res){
        // 为每个检测框生成随机颜色
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        // 绘制边界框
        cv::rectangle(img, re.box, color, 3);
        // 置信度分数
        float confidence = floor(100 * re.confidence) / 100;
        std::cout << std::fixed << std::setprecision(2);
        // 创建标签文本：类别名 + 置信度
        std::string label = p->classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
        // 绘制标签背景矩形
        cv::rectangle(
            img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );
        // 绘制标签文本
        cv::putText(
            img,
            label,
            cv::Point(re.box.x, re.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 0, 0),
            2
        );
    } 
    cv::imshow("Result of Detection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/**
 * 目标检测测试函数：创建检测器并运行示例
 */
void DetectTest(PARAMS params){
    // 创建检测器
    MODEL* Detector = new MODEL;
    // 创建会话，加载模型
    Detector->CreateSession(params);
    // 检测
    Detect(Detector);
    delete Detector;
}

int main(){
    // 图片路径在Detect中修改
    // 设置初始化参数
    PARAMS params;
    params.rectConfidenceThreshold = 0.1;  // 置信度阈值
    params.iouThreshold = 0.5;            // IoU阈值
    params.modelPath = "../model/best.onnx";    // 模型路径
    params.imgSize = { 640, 640 };        // 输入图像尺寸
    params.classes = { "Raw_Banana","Raw_Mango","Ripe_Banana","Ripe_Mango" }; // 类别名称
    DetectTest(params);
    return 0;
}