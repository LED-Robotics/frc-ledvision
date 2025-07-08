#pragma once

#include "common.hpp"
#include <fstream>
using namespace det;

class YOLO11 {
public:
  explicit YOLO11() {};
  ~YOLO11() {};

  virtual void make_pipe(bool warmup = true) {};
  virtual void copy_from_Mat(const cv::Mat &image) {};
  virtual void copy_from_Mat(const cv::Mat &image, cv::Size &size) {};
  virtual void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {};
  virtual void infer() {};
  /*void                 detectPostprocess(std::vector<BoxObject>& objs);*/

  virtual void detectPostprocess(std::vector<BoxObject> &objs,
                         float score_thres = 0.25f, float iou_thres = 0.65f,
                         int topk = 100) {};
  virtual void posePostprocess(std::vector<PoseObject> &objs, float score_thres = 0.25f,
                       float iou_thres = 0.65f, int topk = 100) {};
  // static void          draw_objects(const cv::Mat& image,
  //                                   cv::Mat& res, const
  //                                   std::vector<BoxObject>& objs, const
  //                                   std::vector<std::string>& CLASS_NAMES,
  //                                   const std::vector<std::vector<unsigned
  //                                   int>>& COLORS);
  int num_bindings;
  int num_inputs = 0;
  int num_outputs = 0;
  std::vector<void *> host_ptrs;
  std::vector<void *> device_ptrs;

  PreParam pparam;
};
