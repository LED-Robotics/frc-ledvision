#pragma once

#include "common.hpp"
#include "yolo11.hpp"
#include "onnxruntime_cxx_api.h"
#include <fstream>
using namespace det;

class YOLO11_ONNX : public YOLO11 {
public:
  explicit YOLO11_ONNX(const std::string &model_file_path);
  ~YOLO11_ONNX();

  void make_pipe(bool warmup = true);
  void copy_from_Mat(const cv::Mat &image);
  void copy_from_Mat(const cv::Mat &image, cv::Size &size);
  void infer();
  /*void                 detectPostprocess(std::vector<BoxObject>& objs);*/

  void detectPostprocess(std::vector<BoxObject> &objs,
                         float score_thres = 0.25f, float iou_thres = 0.65f,
                         int topk = 100);
  void posePostprocess(std::vector<PoseObject> &objs, float score_thres = 0.25f,
                       float iou_thres = 0.65f, int topk = 100);
  // static void          draw_objects(const cv::Mat& image,
  //                                   cv::Mat& res, const
  //                                   std::vector<BoxObject>& objs, const
  //                                   std::vector<std::string>& CLASS_NAMES,
  //                                   const std::vector<std::vector<unsigned
  //                                   int>>& COLORS);
  int num_bindings;
  int num_inputs = 0;
  int num_outputs = 0;
  // std::vector<Binding> input_bindings;
  // std::vector<Binding> output_bindings;
  std::vector<void *> host_ptrs;
  std::vector<void *> device_ptrs;

private:
  Ort::Env env;
  Ort::Session* session;
  Ort::RunOptions options;
  std::vector<const char*> inputNodeNames;
  std::vector<const char*> outputNodeNames;

  std::vector<int> imgSize;
  float rectConfidenceThreshold;
  float iouThreshold;
  float resizeScales;//letterbox scale
};
