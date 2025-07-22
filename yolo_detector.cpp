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

#define clock_                           // �������ܲ��Լ�ʱ
#define min(a, b) (((a) < (b)) ? (a) : (b))

// ģ�ͳ�ʼ������
typedef struct PARAMS{
    std::string modelPath;                 // ģ���ļ�·��
    std::vector<int> imgSize = {640, 640}; // ����ͼ��ߴ磬Ĭ��Ϊ 640x640
    float rectConfidenceThreshold = 0.6;   // Ŀ�������Ŷ���ֵ�����ڴ�ֵ�Ŀ�ᱻ����
    float iouThreshold = 0.5;              // IoU��ֵ�����ڷǼ���ֵ����(NMS)
    int keyPointsNum = 2;                  // �ؼ���������������̬���ģ��
    int logSeverityLevel = 3;              // ��־����
    int intraOpNumThreads = 1;             // �ڲ����в������߳���
    std::vector<std::string> classes{
        "Raw_Banana","Raw_Mango","Ripe_Banana","Ripe_Mango"
    };    // ���
} PARAMS;

// �������ṹ�壬���ڴ洢��⡢�������̬���Ľ��
typedef struct RESULT{
    int classId;                        // ���ID
    float confidence;                   // ���Ŷȷ���
    cv::Rect box;                       // �߽�򣨼������λ�ã�
} RESULT;

class MODEL{
public:
    MODEL();
    ~MODEL();
public:
    // ����ONNX�Ự��������ģ��
    void CreateSession(PARAMS &iParams);
    // ִ��������������ͼ�񲢷��ؽ��
    void RunSession(cv::Mat &iImg, std::vector<RESULT> &oResult);
    // Ԥ�ȣ������״�����ģ��ʱ�������Ż�
    void WarmUpSession();
    // ����
    void TensorProcess(clock_t &starttime_1, cv::Mat &iImg, float* &blob, std::vector<int64_t> &inputNodeDims,
                        std::vector<RESULT> &oResult);
    // Ԥ����
    void PreProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg);
    // �������
    std::vector<std::string> classes{};

private:
    Ort::Env env;                              // ONNX Runtime����
    Ort::Session *session;                     // ONNX�Ự������ִ��ģ������
    Ort::RunOptions options;                   // ����ѡ��
    std::vector<const char *> inputNodeNames;  // ����ڵ�����
    std::vector<const char *> outputNodeNames; // ����ڵ�����

    std::vector<int> imgSize;      // ͼ��ߴ�
    float rectConfidenceThreshold; // ���Ŷ���ֵ
    float iouThreshold;            // IoU��ֵ
    float resizeScales;            // ���ű���
};

MODEL::MODEL(){}
MODEL::~MODEL(){delete session;}

/**
 * ��OpenCVͼ��ת��Ϊģ�����������Blob��ʽ
 * @param iImg ����ͼ��
 * @param iBlob ���Blob
 */

void BlobFromImage(cv::Mat &iImg, float* &iBlob){
    int channels = iImg.channels(); // ͨ����
    int imgHeight = iImg.rows;      // �߶�
    int imgWidth = iImg.cols;       // ���

    // ���������Blob��ͬʱ���й�һ��
    for (int c = 0; c < channels; c++)
        for (int h = 0; h < imgHeight; h++)
            for (int w = 0; w < imgWidth; w++)
                // ��ͼ������ת��Ϊ����������һ����[0,1]��Χ
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] =
                 float((iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
}

/**
 * ͼ��Ԥ��������������С��ת����ɫ�ռ��
 * @param iImg ����ԭʼͼ��
 * @param iImgSize Ŀ��ͼ��ߴ�
 * @param oImg ���Ԥ������ͼ��
 */
void MODEL::PreProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg){
    if (iImg.channels() == 3)
        cv::cvtColor(iImg, oImg, cv::COLOR_BGR2RGB);
    else if (iImg.channels() == 1)
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);

    // LetterBox����
    if (iImg.cols >= iImg.rows){
        resizeScales = iImg.cols / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
    }
    else{
        resizeScales = iImg.rows / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
    }
    // ����Ŀ��ߴ�ĺ�ɫͼ�񣬽��������ͼ��������У���Ӻڱߣ�
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;
}

/**
 * ����ONNX�Ự������ģ��
 * @param iParams ��ʼ������
 */
void MODEL::CreateSession(PARAMS& iParams) {
    // ������������Ա����
    rectConfidenceThreshold = iParams.rectConfidenceThreshold;
    iouThreshold = iParams.iouThreshold;
    imgSize = iParams.imgSize;
    classes = iParams.classes;
    const char* modelPath = iParams.modelPath.c_str();
    // ����ONNX Runtime����
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "MODEL");
    Ort::SessionOptions sessionOption;

    // ����ͼ�Ż�������߳���
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // ���������Ż�
    sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads); // �����ڲ������߳���
    sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel); // ������־����

    // ����ONNX�Ự������ģ��
    session = new Ort::Session(env, modelPath, sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    
    // ��ȡģ�͵�����ڵ�����
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    
    // ��ȡģ�͵�����ڵ�����
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++){
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    
    // ��������ѡ�Ԥ�ȻỰ
    options = Ort::RunOptions{ nullptr };
    WarmUpSession();
}

/**
 * ��������Ự
 * @param iImg ����ͼ��
 * @param oResult ���������
 */
void MODEL::RunSession(cv::Mat& iImg, std::vector<RESULT>& oResult) {
#ifdef clock_
    clock_t starttime_1 = clock();  // ��ʼ��ʱ���������ܲ���
#endif

    cv::Mat processedImg;
    // Ԥ��������ͼ��
    PreProcess(iImg, imgSize, processedImg);

    float* blob = new float[processedImg.total() * 3];
    BlobFromImage(processedImg, blob);  // ��������ͼ��ת��ΪBlob
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };  // ��������ά��
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);  // ִ������ͺ���

}

/**
 * �����������ݲ�ִ������ͺ���
 * @param starttime_1 ��ʼʱ�䣬�������ܲ���
 * @param iImg ԭʼ����ͼ��
 * @param blob Ԥ������Blob����
 * @param inputNodeDims ����ڵ�ά��
 * @param oResult ������
 */

void MODEL::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, float* & blob, std::vector<int64_t>& inputNodeDims,
    std::vector<RESULT>& oResult) {
    
    // ������������
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 
        3 * imgSize.at(0) * imgSize.at(1),inputNodeDims.data(), inputNodeDims.size());
        
#ifdef clock_
    clock_t starttime_2 = clock();  // ��¼Ԥ�������ʱ��
#endif // clock_

    // ִ��ģ������
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
        
#ifdef clock_
    clock_t starttime_3 = clock();  // ��¼�������ʱ��
#endif // clock_

    // ��ȡ���������Ϣ
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<float>();
    
    // �ͷ�Blob�ڴ�
    delete[] blob;
    
    int signalResultNum = outputNodeDims[1];  // ÿ�������������
    int strideNum = outputNodeDims[2];        // ���������
    
    // �洢�������ݵ�����
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // ���������ת��ΪOpenCV�����Ա㴦��
    cv::Mat rawData;

    rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);

    // ת�þ���
    rawData = rawData.t();
    float* data = (float*)rawData.data;

    // ����ÿ������
    for (int i = 0; i < strideNum; ++i){
        float* classesScores = data + 4;  // ǰ4��������
        cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        
        // �ҳ���߷��������
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        
        // �����߷���������ֵ���򱣴��������
        if (maxClassScore > rectConfidenceThreshold){
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            
            // �������꣨������ĵ�����Ϳ�ߣ�
            float x = data[0];  // ����x����
            float y = data[1];  // ����y����
            float w = data[2];  // ���
            float h = data[3];  // �߶�

            // ת��Ϊ���Ͻ�����Ϳ�߸�ʽ����Ӧ�����ű���
            int left = int((x - 0.5 * w) * resizeScales);
            int top = int((y - 0.5 * h) * resizeScales);
            int width = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // �ƶ�����һ����������
        data += signalResultNum;
    }
    
    // NMS�����ص��Ŀ�
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
    
    // �������ս��
    for (int i = 0; i < nmsResult.size(); ++i){
        int idx = nmsResult[i];
        RESULT result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        oResult.push_back(result);
    }

#ifdef clock_
    // ���㲢�������ָ��
    clock_t starttime_4 = clock();
    double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
    std::cout << "[MODEL]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
#endif

}

/**
 * Ԥ�ȻỰ�����������Ż��״���������
 * ����һ���հ�ͼ��ִ��һ�������Ա�ONNX Runtime��ɳ�ʼ��
 */
void MODEL::WarmUpSession() {
    clock_t starttime_1 = clock();
    
    // ����һ���հ�ͼ��
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    
    // Ԥ����ͼ��
    PreProcess(iImg, imgSize, processedImg);
    
    float* blob = new float[iImg.total() * 3];
    BlobFromImage(processedImg, blob);
    std::vector<int64_t> input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        input_node_dims.data(), input_node_dims.size());// ������������
    auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
        outputNodeNames.size()); // ִ������
        
    delete[] blob;
    clock_t starttime_4 = clock();
    double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
}

/**
 * Ŀ������ʾ��������ȡimagesĿ¼�µ�ͼƬ�����м��ģ�ͣ�����ʾ���
 * @param p MODEL����ָ��
 */
void Detect(MODEL*& p) {
    std::string img_path = "../images/Raw_Mango_0_5537.jpg";
    cv::Mat img = cv::imread(img_path);
    std::vector<RESULT> res;
    // ��ȡ�����
    p->RunSession(img, res);
    // ����ÿ�������
    for (auto& re : res){
        // Ϊÿ���������������ɫ
        cv::RNG rng(cv::getTickCount());
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        // ���Ʊ߽��
        cv::rectangle(img, re.box, color, 3);
        // ���Ŷȷ���
        float confidence = floor(100 * re.confidence) / 100;
        std::cout << std::fixed << std::setprecision(2);
        // ������ǩ�ı�������� + ���Ŷ�
        std::string label = p->classes[re.classId] + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);
        // ���Ʊ�ǩ��������
        cv::rectangle(
            img,
            cv::Point(re.box.x, re.box.y - 25),
            cv::Point(re.box.x + label.length() * 15, re.box.y),
            color,
            cv::FILLED
        );
        // ���Ʊ�ǩ�ı�
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
 * Ŀ������Ժ��������������������ʾ��
 */
void DetectTest(PARAMS params){
    // ���������
    MODEL* Detector = new MODEL;
    // �����Ự������ģ��
    Detector->CreateSession(params);
    // ���
    Detect(Detector);
    delete Detector;
}

int main(){
    // ͼƬ·����Detect���޸�
    // ���ó�ʼ������
    PARAMS params;
    params.rectConfidenceThreshold = 0.1;  // ���Ŷ���ֵ
    params.iouThreshold = 0.5;            // IoU��ֵ
    params.modelPath = "../model/best.onnx";    // ģ��·��
    params.imgSize = { 640, 640 };        // ����ͼ��ߴ�
    params.classes = { "Raw_Banana","Raw_Mango","Ripe_Banana","Ripe_Mango" }; // �������
    DetectTest(params);
    return 0;
}