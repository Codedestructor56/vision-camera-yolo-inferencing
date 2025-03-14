#include "onnxFrameProcessor.h"
#include "TypedArray.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <sstream>
#include <cmath>

#ifdef HAVE_OPENCL
#include <opencv2/core/ocl.hpp>
#endif

#if defined(__ANDROID_API__) && (__ANDROID_API__ >= 29)
#include <android/NeuralNetworks.h>
#endif

using namespace facebook;
using namespace jsi;


OnnxFrameProcessor::OnnxFrameProcessor()
    : modelLoaded(false), accelerationSet(false) {
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Processor created");
}

OnnxFrameProcessor::~OnnxFrameProcessor() {
  clearState();
}

void OnnxFrameProcessor::clearState() {
  net = cv::dnn::Net();
  currentModelPath.clear();
  currentModelType.clear();
  modelLoaded = false;
  accelerationSet = false;
}

void OnnxFrameProcessor::configureDevice(const std::string &modelType) {
  accelerationSet = false;
#ifdef HAVE_OPENCL
  if (cv::ocl::haveOpenCL()) {
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "OpenCL available. Device: %s", device.name().c_str());
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    accelerationSet = true;
  }
#endif

#if defined(__ANDROID_API__) && (__ANDROID_API__ >= 29)
  if (!accelerationSet && modelType == "tflite") {
    uint32_t deviceCount = 0;
    int ret = ANeuralNetworks_getDeviceCount(&deviceCount);
    if (ret == ANEURALNETWORKS_NO_ERROR && deviceCount > 0) {
      __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "NNAPI available with %u devices", deviceCount);
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
      accelerationSet = true;
    } else {
      __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "No NNAPI devices found (ret=%d, count=%u)", ret, deviceCount);
    }
  }
#else
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Android API < 29, NNAPI query not available");
#endif

  if (!accelerationSet) {
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "No hardware acceleration available. Falling back to CPU.");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU_FP16);
  }
}

void OnnxFrameProcessor::loadModel(const std::string &modelPath, const std::string &modelType) {
  if (modelLoaded && currentModelPath == modelPath && currentModelType == modelType) {
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Model already loaded");
    return;
  }
  currentModelPath = modelPath;
  currentModelType = modelType;
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Loading model from: %s", modelPath.c_str());

  if (modelType == "onnx") {
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Failed to load ONNX model");
      throw std::runtime_error("Failed to load ONNX model");
    }
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "ONNX model loaded successfully");
  } else if (modelType == "tflite") {
    net = cv::dnn::readNetFromTFLite(modelPath);
    if (net.empty()) {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Failed to load TFLite model");
      throw std::runtime_error("Failed to load TFLite model");
    }
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "TFLite model loaded successfully");
  } else {
    __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Invalid model type: %s", modelType.c_str());
    throw std::runtime_error("Invalid model type, expected 'tflite' or 'onnx'");
  }
  configureDevice(modelType);
  modelLoaded = true;
}

std::vector<std::string> OnnxFrameProcessor::processFrame(const cv::Mat &image,
                                                          const std::vector<std::string> &classes,
                                                          float modelConfidenceThreshold,
                                                          float modelScoreThreshold,
                                                          float modelNMSThreshold) {
  if (!modelLoaded) {
    __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Model not loaded");
    throw std::runtime_error("Model not loaded");
  }

  int origHeight = image.rows;
  int origWidth = image.cols;
  int length = std::max(origWidth, origHeight);
  float ratio = std::min(length / float(origWidth), length / float(origHeight));
  int newUnpadWidth = static_cast<int>(std::round(origWidth * ratio));
  int newUnpadHeight = static_cast<int>(std::round(origHeight * ratio));
  int dw = static_cast<int>(std::ceil((length - newUnpadWidth) / 2.0));
  int dh = static_cast<int>(std::ceil((length - newUnpadHeight) / 2.0));

  cv::Mat resizedImg;
  cv::resize(image, resizedImg, cv::Size(newUnpadWidth, newUnpadHeight), 0, 0, cv::INTER_LINEAR);

  cv::Mat paddedImg;
  cv::copyMakeBorder(resizedImg, paddedImg, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

  cv::Mat blob;
  cv::dnn::blobFromImage(paddedImg, blob, 1/255.0, cv::Size(length, length), cv::Scalar(), true, false);
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Input blob created, size: %dx%d", blob.size[2], blob.size[3]);

  return runInference(blob, classes, modelConfidenceThreshold, modelScoreThreshold, modelNMSThreshold);
}

std::vector<std::string> OnnxFrameProcessor::runInference(const cv::Mat &blob,
                                                          const std::vector<std::string> &classes,
                                                          float modelConfidenceThreshold,
                                                          float modelScoreThreshold,
                                                          float modelNMSThreshold) {
  net.setInput(blob);
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Input blob set");
  std::vector<cv::Mat> outputs;
  double t0 = cv::getTickCount();
  net.forward(outputs, net.getUnconnectedOutLayersNames());
  double t1 = cv::getTickCount();
  double inferenceTime = (t1 - t0) * 1000 / cv::getTickFrequency();
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Forward pass completed in %.2f ms", inferenceTime);

  if (outputs.empty()) {
    __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "No output from network");
    throw std::runtime_error("Empty network output");
  }

  int rows_out = outputs[0].size[1];
  int dimensions = outputs[0].size[2];
  bool yolov8 = false;
  if (dimensions > rows_out) {
    rows_out = outputs[0].size[2];
    dimensions = outputs[0].size[1];
    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);
    yolov8 = true;
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Reshaped output for yolov8 configuration");
  }

  float* data = reinterpret_cast<float*>(outputs[0].data);
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows_out; ++i) {
    if (yolov8) {
      float* classes_scores = data + 4;
      cv::Mat scores(1, static_cast<int>(classes.size()), CV_32FC1, classes_scores);
      cv::Point class_id_point;
      double maxClassScore;
      cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id_point);
      if (maxClassScore > modelScoreThreshold) {
        confidences.push_back(maxClassScore);
        class_ids.push_back(class_id_point.x);
        float x = data[0], y = data[1], w = data[2], h = data[3];
        boxes.push_back(cv::Rect(cv::Point(x, y), cv::Size(w, h)));
      }
    } else {
      float confidence = data[4];
      if (confidence >= modelConfidenceThreshold) {
        float* classes_scores = data + 5;
        cv::Mat scores(1, static_cast<int>(classes.size()), CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
        if (max_class_score > modelScoreThreshold) {
          confidences.push_back(confidence);
          class_ids.push_back(class_id_point.x);
          float x = data[0], y = data[1], w = data[2], h = data[3];
          boxes.push_back(cv::Rect(cv::Point(x, y), cv::Size(w, h)));
        }
      }
    }
    data += dimensions;
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indices);
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "NMSBoxes reduced detections to %zu", indices.size());

  std::vector<std::string> detections;
  for (size_t i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    std::ostringstream oss;
    oss << "{ \"class_id\": " << class_ids[idx]
        << ", \"class_name\": \"" << getClassName(class_ids[idx], classes) << "\""
        << ", \"confidence\": " << confidences[idx]
        << ", \"box\": [" << boxes[idx].x << ", " << boxes[idx].y
        << ", " << boxes[idx].width << ", " << boxes[idx].height << "] }";
    detections.push_back(oss.str());
  }
  return detections;
}

static std::shared_ptr<OnnxFrameProcessor> gProcessor = std::make_shared<OnnxFrameProcessor>();

void OnnxFrameProcessor::registerOnnxFrameProcessor(jsi::Runtime &runtime) {
  auto onnxProcessorFunc = [=](jsi::Runtime &runtime,
                               const jsi::Value &thisArg,
                               const jsi::Value *args,
                               size_t count) -> jsi::Value {
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Starting processOnnxFrame");
    if (count < 10) {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Expected 10 arguments, received %zu", count);
      throw jsi::JSError(runtime, "Expected 10 arguments");
    }

    double rows = args[0].asNumber();
    double cols = args[1].asNumber();
    double channels = args[2].asNumber();
    jsi::Object input = args[3].asObject(runtime);

    int type = -1;
    if (channels == 1) {
      type = CV_8U;
    } else if (channels == 3) {
      type = CV_8UC3;
    } else if (channels == 4) {
      type = CV_8UC4;
    } else {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Invalid channel count: %f", channels);
      throw jsi::JSError(runtime, "Invalid channel count passed to frameBufferToMat!");
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Image dimensions: %fx%f, channels: %f", rows, cols, channels);
    auto inputBuffer = mrousavy::getTypedArray(runtime, std::move(input));
    auto vec = inputBuffer.toVector(runtime);
    cv::Mat image(rows, cols, type);
    memcpy(image.data, vec.data(), static_cast<size_t>(rows * cols * channels));
    if (image.empty()) {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Image is empty");
      throw jsi::JSError(runtime, "Empty image");
    }

    std::string modelPath = args[4].asString(runtime).utf8(runtime);
    float modelConfidenceThreshold = args[5].asNumber();
    float modelScoreThreshold = args[6].asNumber();
    float modelNMSThreshold = args[7].asNumber();

    jsi::Object classesObject = args[8].asObject(runtime);
    jsi::Value lengthValue = classesObject.getProperty(runtime, "length");
    if (!lengthValue.isNumber())
      throw jsi::JSError(runtime, "Class names array must have a numeric length");
    size_t arrayLength = static_cast<size_t>(lengthValue.asNumber());
    if (arrayLength == 0)
      throw jsi::JSError(runtime, "Class names array cannot be empty");
    jsi::Array classNamesArray = classesObject.asArray(runtime);
    std::vector<std::string> classes;
    for (size_t i = 0; i < arrayLength; i++) {
      jsi::Value val = classNamesArray.getValueAtIndex(runtime, i);
      if (!val.isString())
        throw jsi::JSError(runtime, "Class names array must contain only strings");
      classes.push_back(val.asString(runtime).utf8(runtime));
    }
    std::string modelType = args[9].asString(runtime).utf8(runtime);

    if (!gProcessor->isModelLoaded()) {
       __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Loading model");
      gProcessor->loadModel(modelPath, modelType);
    }

    auto detections = gProcessor->processFrame(image, classes,
                                               modelConfidenceThreshold,
                                               modelScoreThreshold,
                                               modelNMSThreshold);

    jsi::Array jsiDetections(runtime, detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
      jsiDetections.setValueAtIndex(runtime, i,
          jsi::Value(runtime, jsi::String::createFromUtf8(runtime, detections[i].c_str())));
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Returning %zu detections", detections.size());
    return jsiDetections;
  };

  auto func = jsi::Function::createFromHostFunction(runtime,
                 jsi::PropNameID::forUtf8(runtime, "processOnnxFrame"),
                 10,
                 onnxProcessorFunc);
  runtime.global().setProperty(runtime, "processOnnxFrame", func);
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "processOnnxFrame registered");
}
