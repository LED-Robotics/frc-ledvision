#include "yolo11-onnx.hpp"
#include <regex>

YOLO11_ONNX::YOLO11_ONNX(const std::string &model_file_path) {
  char *Ret = nullptr;
  std::regex pattern("[\u4e00-\u9fa5]");
  bool result = std::regex_search(model_file_path, pattern);
  try {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
    Ort::SessionOptions sessionOption;
    sessionOption.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetLogSeverityLevel(3);

    const char *modelPath = model_file_path.c_str();

    session = new Ort::Session(env, modelPath, sessionOption);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
      Ort::AllocatedStringPtr input_node_name =
          session->GetInputNameAllocated(i, allocator);
      char *temp_buf = new char[50];
      strcpy(temp_buf, input_node_name.get());
      inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
      Ort::AllocatedStringPtr output_node_name =
          session->GetOutputNameAllocated(i, allocator);
      char *temp_buf = new char[10];
      strcpy(temp_buf, output_node_name.get());
      outputNodeNames.push_back(temp_buf);
    }
    options = Ort::RunOptions{nullptr};
  } catch (const std::exception &e) {
    const char *str1 = "[YOLO_V11]:";
    const char *str2 = e.what();
    std::string result = std::string(str1) + std::string(str2);
    char *merged = new char[result.length() + 1];
    std::strcpy(merged, result.c_str());
    std::cout << merged << std::endl;
    delete[] merged;
  }
}

YOLO11_ONNX::~YOLO11_ONNX() { delete session; }

void YOLO11_ONNX::make_pipe(bool warmup) {
  
}

void YOLO11_ONNX::copy_from_Mat(const cv::Mat &image) {}

void YOLO11_ONNX::copy_from_Mat(const cv::Mat &image, cv::Size &size) {}

void YOLO11_ONNX::infer() {}

void YOLO11_ONNX::detectPostprocess(std::vector<BoxObject> &objs,
                                    float score_thres, float iou_thres,
                                    int topk) {}

void YOLO11_ONNX::posePostprocess(std::vector<PoseObject> &objs,
                                  float score_thres, float iou_thres,
                                  int topk) {}
