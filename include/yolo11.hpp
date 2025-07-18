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
  void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
      cv::resize(image, tmp, cv::Size(padw, padh));
    } else {
      tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                       {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0),
                           true, false, CV_32F);
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
  };
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
