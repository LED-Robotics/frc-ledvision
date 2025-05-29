#include "yolo11.hpp"

YOLO11::YOLO11(const std::string &engine_file_path) {
  std::ifstream file(engine_file_path, std::ios::binary);
  assert(file.good());
  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *trtModelStream = new char[size];
  assert(trtModelStream);
  file.read(trtModelStream, size);
  file.close();
  initLibNvInferPlugins(&this->gLogger, "");
  this->runtime = nvinfer1::createInferRuntime(this->gLogger);
  assert(this->runtime != nullptr);

  this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
  assert(this->engine != nullptr);
  delete[] trtModelStream;
  this->context = this->engine->createExecutionContext();

  assert(this->context != nullptr);
  cudaStreamCreate(&this->stream);
  this->num_bindings = this->engine->getNbIOTensors();

  for (int i = 0; i < this->num_bindings; i++) {
    Binding binding;
    nvinfer1::Dims dims;
    const char *name = this->engine->getIOTensorName(i);
    nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
    binding.name = name;
    binding.dsize = type_to_size(dtype);

    bool IsInput =
        engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    if (IsInput) {
      this->num_inputs += 1;
      dims = this->engine->getProfileShape(name, 0,
                                           nvinfer1::OptProfileSelector::kMAX);
      binding.size = get_size_by_dims(dims);
      binding.dims = dims;
      this->input_bindings.push_back(binding);
      // set max opt shape
      this->context->setInputShape(name, dims);
    } else {
      dims = this->context->getTensorShape(name);
      binding.size = get_size_by_dims(dims);
      binding.dims = dims;
      this->output_bindings.push_back(binding);
      this->num_outputs += 1;
    }
  }
}

YOLO11::~YOLO11() {
  delete this->context;
  delete this->engine;
  delete this->runtime;
  cudaStreamDestroy(this->stream);
  for (auto &ptr : this->device_ptrs) {
    CHECK(cudaFree(ptr));
  }

  for (auto &ptr : this->host_ptrs) {
    CHECK(cudaFreeHost(ptr));
  }
}
void YOLO11::make_pipe(bool warmup) {

  for (auto &bindings : this->input_bindings) {
    void *d_ptr;
    CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
    this->context->setTensorAddress(bindings.name.c_str(), d_ptr);
    this->device_ptrs.push_back(d_ptr);
  }

  for (auto &bindings : this->output_bindings) {
    void *d_ptr, *h_ptr;
    size_t size = bindings.size * bindings.dsize;
    CHECK(cudaMalloc(&d_ptr, size));
    CHECK(cudaHostAlloc(&h_ptr, size, 0));
    this->context->setTensorAddress(bindings.name.c_str(), d_ptr);
    this->device_ptrs.push_back(d_ptr);
    this->host_ptrs.push_back(h_ptr);
  }

  if (warmup) {
    for (int i = 0; i < 10; i++) {
      for (auto &bindings : this->input_bindings) {
        size_t size = bindings.size * bindings.dsize;
        void *h_ptr = malloc(size);
        memset(h_ptr, 0, size);
        CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size,
                              cudaMemcpyHostToDevice, this->stream));
        free(h_ptr);
      }
      this->infer();
    }
    printf("model warmup 10 times\n");
  }
}

bool YOLO11::generateEngine(std::string onnxPath) {

  Logger logger{nvinfer1::ILogger::Severity::kERROR};
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
  nvonnxparser::IParser *parser = createParser(*network, logger);
  bool result = parser->parseFromFile(
      onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  std::cout << "Beginning engine generation of model: " << onnxPath << "..."
            << std::endl;
  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  nvinfer1::IHostMemory *serializedModel =
      builder->buildSerializedNetwork(*network, *config);
  std::string enginePath = onnxPath.substr(0, onnxPath.size() - 4) + "engine";
  std::ofstream file;
  // Open the engine file in binary write mode
  file.open(enginePath, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    std::cout << "Create engine file " << enginePath << " failed" << std::endl;
    return false;
  }
  // Write the serialized engine data to the file
  file.write((const char *)serializedModel->data(), serializedModel->size());
  file.close();

  std::cout << "Engine successfully saved as: " << enginePath << std::endl;

  delete network;
  delete config;
  delete builder;
  delete parser;

  return true;
}

void YOLO11::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
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
}

void YOLO11::copy_from_Mat(const cv::Mat &image) {
  cv::Mat nchw;
  auto &in_binding = this->input_bindings[0];
  int width = in_binding.dims.d[3];
  int height = in_binding.dims.d[2];
  cv::Size size{width, height};
  this->letterbox(image, nchw, size);

  this->context->setInputShape(in_binding.name.c_str(),
                               nvinfer1::Dims{4, {1, 3, height, width}});

  CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                        nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                        this->stream));
}

void YOLO11::copy_from_Mat(const cv::Mat &image, cv::Size &size) {
  cv::Mat nchw;
  auto &in_binding = this->input_bindings[0];
  this->letterbox(image, nchw, size);
  this->context->setInputShape(
      in_binding.name.c_str(),
      nvinfer1::Dims{4, {1, 3, size.height, size.width}});
  CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                        nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                        this->stream));
}

void YOLO11::infer() {
  this->context->enqueueV3(this->stream);
  for (int i = 0; i < this->num_outputs; i++) {
    size_t osize =
        this->output_bindings[i].size * this->output_bindings[i].dsize;
    CHECK(cudaMemcpyAsync(this->host_ptrs[i],
                          this->device_ptrs[i + this->num_inputs], osize,
                          cudaMemcpyDeviceToHost, this->stream));
  }
  cudaStreamSynchronize(this->stream);
}

void YOLO11::detectPostprocess(std::vector<BoxObject> &objs, float score_thres,
                               float iou_thres, int topk) {
  objs.clear();
  auto num_channels = this->output_bindings[0].dims.d[1];
  auto num_anchors = this->output_bindings[0].dims.d[2];

  auto &dw = this->pparam.dw;
  auto &dh = this->pparam.dh;
  auto &width = this->pparam.width;
  auto &height = this->pparam.height;
  auto &ratio = this->pparam.ratio;

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;

  cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                           static_cast<float *>(this->host_ptrs[0]));
  output = output.t();
  for (int i = 0; i < num_anchors; i++) {
    auto row = output.row(i);
    auto row_ptr = row.ptr<float>();
    auto bboxes_ptr = row_ptr;
    auto scores_ptr = row_ptr + 4;
    cv::Mat score_mat(1, row.cols - 4, CV_32FC1, scores_ptr);
    cv::Point class_id;
    double maxClassScore;

    cv::minMaxLoc(score_mat, NULL, &maxClassScore, NULL, &class_id);

    if (maxClassScore > score_thres) {
      float x = *bboxes_ptr++ - dw;
      float y = *bboxes_ptr++ - dh;
      float w = *bboxes_ptr++;
      float h = *bboxes_ptr;

      float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
      float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
      float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
      float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

      cv::Rect_<float> bbox;
      bbox.x = x0;
      bbox.y = y0;
      bbox.width = x1 - x0;
      bbox.height = y1 - y0;

      bboxes.push_back(bbox);
      labels.push_back(class_id.x);
      scores.push_back(maxClassScore);
    }
  }

#ifdef BATCHED_NMS
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres,
                           indices);
#else
  cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

  int cnt = 0;
  for (auto &i : indices) {
    if (cnt >= topk) {
      break;
    }
    BoxObject obj;
    obj.rect = bboxes[i];
    obj.prob = scores[i];
    obj.label = labels[i];
    objs.push_back(obj);
    cnt += 1;
  }
}

void YOLO11::posePostprocess(std::vector<PoseObject> &objs, float score_thres,
                             float iou_thres, int topk) {
  objs.clear();
  auto num_channels = this->output_bindings[0].dims.d[1];
  auto num_anchors = this->output_bindings[0].dims.d[2];

  auto &dw = this->pparam.dw;
  auto &dh = this->pparam.dh;
  auto &width = this->pparam.width;
  auto &height = this->pparam.height;
  auto &ratio = this->pparam.ratio;

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;
  std::vector<std::vector<float>> kpss;

  cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                           static_cast<float *>(this->host_ptrs[0]));
  output = output.t();
  for (int i = 0; i < num_anchors; i++) {
    auto row_ptr = output.row(i).ptr<float>();
    auto bboxes_ptr = row_ptr;
    auto scores_ptr = row_ptr + 4;
    auto kps_ptr = row_ptr + 5;

    float score = *scores_ptr;
    if (score > score_thres) {
      float x = *bboxes_ptr++ - dw;
      float y = *bboxes_ptr++ - dh;
      float w = *bboxes_ptr++;
      float h = *bboxes_ptr;

      float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
      float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
      float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
      float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

      cv::Rect_<float> bbox;
      bbox.x = x0;
      bbox.y = y0;
      bbox.width = x1 - x0;
      bbox.height = y1 - y0;
      std::vector<float> kps;
      for (int k = 0; k < 17; k++) {
        float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
        float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
        float kps_s = *(kps_ptr + 3 * k + 2);
        kps_x = clamp(kps_x, 0.f, width);
        kps_y = clamp(kps_y, 0.f, height);
        kps.push_back(kps_x);
        kps.push_back(kps_y);
        kps.push_back(kps_s);
      }

      bboxes.push_back(bbox);
      labels.push_back(0);
      scores.push_back(score);
      kpss.push_back(kps);
    }
  }

#ifdef BATCHED_NMS
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres,
                           indices);
#else
  cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

  int cnt = 0;
  for (auto &i : indices) {
    if (cnt >= topk) {
      break;
    }
    PoseObject obj;
    obj.rect = bboxes[i];
    obj.prob = scores[i];
    obj.label = labels[i];
    obj.kps = kpss[i];
    objs.push_back(obj);
    cnt += 1;
  }
}
