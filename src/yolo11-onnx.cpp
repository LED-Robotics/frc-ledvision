#include "yolo11-onnx.hpp"

YOLO11_ONNX::YOLO11_ONNX(const std::string &engine_file_path) {

}

YOLO11_ONNX::~YOLO11_ONNX() {
  
}
void YOLO11_ONNX::make_pipe(bool warmup) {
  
}

bool YOLO11_ONNX::generateEngine(std::string onnxPath) {

}

void YOLO11_ONNX::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {

}

void YOLO11_ONNX::copy_from_Mat(const cv::Mat &image) {

}

void YOLO11_ONNX::copy_from_Mat(const cv::Mat &image, cv::Size &size) {

}

void YOLO11_ONNX::infer() {

}

void YOLO11_ONNX::detectPostprocess(std::vector<BoxObject> &objs, float score_thres,
                               float iou_thres, int topk) {
}

void YOLO11_ONNX::posePostprocess(std::vector<PoseObject> &objs, float score_thres,
                             float iou_thres, int topk) {
}
