#pragma once

#ifdef CUDA_PRESENT
#include "NvInfer.h"
#endif
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>

#ifdef CUDA_PRESENT

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                 \
      printf("    File:       %s\n", __FILE__);                                \
      printf("    Line:       %d\n", __LINE__);                                \
      printf("    Error code: %d\n", error_code);                              \
      printf("    Error text: %s\n", cudaGetErrorString(error_code));          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

class Logger : public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity reportableSeverity;

  explicit Logger(
      nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
      : reportableSeverity(severity) {}

  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    if (severity > reportableSeverity) {
      return;
    }
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      std::cerr << "INTERNAL_ERROR: ";
      break;
    case nvinfer1::ILogger::Severity::kERROR:
      std::cerr << "ERROR: ";
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      std::cerr << "WARNING: ";
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      std::cerr << "INFO: ";
      break;
    default:
      std::cerr << "VERBOSE: ";
      break;
    }
    std::cerr << msg << std::endl;
  }
};

inline int get_size_by_dims(const nvinfer1::Dims &dims) {
  int size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }
  return size;
}

inline int type_to_size(const nvinfer1::DataType &dataType) {
  switch (dataType) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
    break;
  case nvinfer1::DataType::kHALF:
    return 2;
    break;
  case nvinfer1::DataType::kFP8:
    return 1;
    break;
  case nvinfer1::DataType::kBF16:
    return 2;
    break;
  case nvinfer1::DataType::kINT32:
    return 4;
    break;
  case nvinfer1::DataType::kINT64:
    return 8;
    break;
  case nvinfer1::DataType::kINT8:
    return 1;
    break;
  case nvinfer1::DataType::kUINT8:
    return 1;
    break;
  case nvinfer1::DataType::kBOOL:
    return 1;
    break;
  default:
    return 4;
    break;
  }
}

#endif

inline static float clamp(float val, float min, float max) {
  return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string &path) {
  if (access(path.c_str(), 0) == F_OK) {
    return true;
  }
  return false;
}

inline bool IsFile(const std::string &path) {
  if (!IsPathExist(path)) {
    printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
    return false;
  }
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string &path) {
  if (!IsPathExist(path)) {
    return false;
  }
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

namespace det {
#ifdef CUDA_PRESENT
struct Binding {
  size_t size = 1;
  size_t dsize = 1;
  nvinfer1::Dims dims;
  std::string name;
};
#endif // CUDA_PRESENT

struct BoxObject {
  cv::Rect_<float> rect;
  int label = 0;
  float prob = 0.0;
};

struct PoseObject {
  cv::Rect_<float> rect;
  int label = 0;
  float prob = 0.0;
  std::vector<float> kps;
};

enum DetectionTypes { Box, Pose };

struct PreParam {
  float ratio = 1.0f;
  float dw = 0.0f;
  float dh = 0.0f;
  float height = 0;
  float width = 0;
};
} // namespace det
