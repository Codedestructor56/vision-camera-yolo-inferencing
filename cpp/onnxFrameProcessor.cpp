#ifdef HAVE_OPENCL
#include <opencv2/core/ocl.hpp>
#endif
#include <jsi/jsi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "Promise.h"
#include "TypedArray.h"
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <android/log.h>
#include <android/NeuralNetworks.h>

using namespace facebook;
using namespace jsi;

std::string getClassName(int id, const std::vector<std::string>& classes) {
  if (id >= 0 && id < classes.size()) return classes[id];
  return "unknown";
}

extern "C" void registerOnnxFrameProcessor(jsi::Runtime &runtime) {
  auto onnxProcessor = [](jsi::Runtime &runtime,
                          const jsi::Value &thisArg,
                          const jsi::Value *args,
                          size_t count) -> jsi::Value {
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Starting processOnnxFrame");
    if (count < 10) {
      __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Expected 10 arguments, received %zu", count);
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
      __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Invalid channel count: %f", channels);
      throw std::runtime_error("Invalid channel count passed to frameBufferToMat!");
    }
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Image dimensions: %fx%f, channels: %f", rows, cols, channels);
    auto inputBuffer = mrousavy::getTypedArray(runtime, std::move(input));
    auto vec = inputBuffer.toVector(runtime);
    cv::Mat image(rows, cols, type);
    memcpy(image.data, vec.data(), static_cast<size_t>(rows * cols * channels));
    if (image.empty()) {
      __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Image is empty");
      throw jsi::JSError(runtime, "Empty image");
    }
    std::string modelPath = args[4].asString(runtime).utf8(runtime);
    std::string modelType = args[9].asString(runtime).utf8(runtime);
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Loading model from: %s", modelPath.c_str());
    cv::dnn::Net net;
    if (modelType == "onnx") {
      net = cv::dnn::readNetFromONNX(modelPath);
      if (net.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Failed to load ONNX model");
        throw jsi::JSError(runtime, "Failed to load ONNX model");
      }
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "ONNX model loaded successfully");
    } else if (modelType == "tflite") {
      net = cv::dnn::readNetFromTFLite(modelPath);
      if (net.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Failed to load TFLite model");
        throw jsi::JSError(runtime, "Failed to load TFLite model");
      }
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "TFLite model loaded successfully");
    } else {
      __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Invalid model type: %s", modelType.c_str());
      throw std::runtime_error("Invalid model type, expected 'tflite' or 'onnx'");
    }
    int origHeight = image.rows;
    int origWidth = image.cols;
    int length = std::max(origWidth, origHeight);
    float ratio = std::min(length / float(origWidth), length / float(origHeight));
    int newUnpadWidth = static_cast<int>(round(origWidth * ratio));
    int newUnpadHeight = static_cast<int>(round(origHeight * ratio));
    int dw = static_cast<int>(std::ceil((length - newUnpadWidth) / 2.0));
    int dh = static_cast<int>(std::ceil((length - newUnpadHeight) / 2.0));
    cv::Mat resizedImg;
    cv::resize(image, resizedImg, cv::Size(newUnpadWidth, newUnpadHeight), cv::INTER_LINEAR);
    cv::Mat paddedImg;
    cv::copyMakeBorder(resizedImg, paddedImg, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob;
    cv::dnn::blobFromImage(paddedImg, blob, 1/255.0, cv::Size(length, length), cv::Scalar(), true, false);
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Input blob created, size: %dx%d", blob.size[2], blob.size[3]);
    net.setInput(blob);

    bool accelerationSet = false;
    if (cv::ocl::haveOpenCL()) {
      cv::ocl::Device device = cv::ocl::Device::getDefault();
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "OpenCL available. Device: %s", device.name().c_str());
    }
#ifdef HAVE_OPENCL
    if (cv::ocl::haveOpenCL()) {
      cv::ocl::Device device = cv::ocl::Device::getDefault();
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "OpenCL available. Device: %s", device.name().c_str());
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
         __android_log_print(ANDROID_LOG_DEBUG, "NNAPI", "NNAPI available with %u devices", deviceCount);
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
         accelerationSet = true;
      } else {
         __android_log_print(ANDROID_LOG_DEBUG, "NNAPI", "No NNAPI devices found (ret=%d, count=%u)", ret, deviceCount);
      }
    }
#else
    __android_log_print(ANDROID_LOG_DEBUG, "NNAPI", "Android API < 29, NNAPI query not available");
#endif

    if (!accelerationSet) {
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "No hardware acceleration available. Falling back to CPU.");
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU_FP16);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Starting forward pass");
    std::vector<cv::Mat> outputs;
    double t0 = cv::getTickCount();
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    double t1 = cv::getTickCount();
    double inferenceTime = (t1 - t0) * 1000 / cv::getTickFrequency();
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Forward pass completed in %.2f ms", inferenceTime);

    int rows_out = outputs[0].size[1];
    int dimensions = outputs[0].size[2];
    bool yolov8 = false;
    if (dimensions > rows_out) {
      rows_out = outputs[0].size[2];
      dimensions = outputs[0].size[1];
      outputs[0] = outputs[0].reshape(1, dimensions);
      cv::transpose(outputs[0], outputs[0]);
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Reshaped output for yolov8 configuration");
    }
    float* data = reinterpret_cast<float*>(outputs[0].data);
    float modelConfidenseThreshold = args[5].asNumber();
    float modelScoreThreshold = args[6].asNumber();
    float modelNMSThreshold = args[7].asNumber();
    jsi::Object classesObject = args[8].asObject(runtime);
    jsi::Value lengthValue = classesObject.getProperty(runtime, "length");
    if (!lengthValue.isNumber()) throw jsi::JSError(runtime, "Class names array must have a numeric length");
    size_t arrayLength = static_cast<size_t>(lengthValue.asNumber());
    if (arrayLength == 0) throw jsi::JSError(runtime, "Class names array cannot be empty");
    jsi::Array classNamesArray = classesObject.asArray(runtime);
    std::vector<std::string> classes;
    for (size_t i = 0; i < arrayLength; i++) {
      jsi::Value val = classNamesArray.getValueAtIndex(runtime, i);
      if (!val.isString()) throw jsi::JSError(runtime, "Class names array must contain only strings");
      classes.push_back(val.asString(runtime).utf8(runtime));
    }
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < rows_out; ++i) {
      if (yolov8) {
        float* classes_scores = data + 4;
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id_point);
        if (maxClassScore > modelScoreThreshold) {
          confidences.push_back(maxClassScore);
          class_ids.push_back(class_id_point.x);
          float x = data[0];
          float y = data[1];
          float w = data[2];
          float h = data[3];
          boxes.push_back(cv::Rect(x, y, w, h));
        }
      } else {
        float confidence = data[4];
        if (confidence >= modelConfidenseThreshold) {
          float* classes_scores = data + 5;
          cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
          cv::Point class_id_point;
          double max_class_score;
          minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
          if (max_class_score > modelScoreThreshold) {
            confidences.push_back(confidence);
            class_ids.push_back(class_id_point.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            boxes.push_back(cv::Rect(x, y, w, h));
          }
        }
      }
      data += dimensions;
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indices);
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "NMSBoxes reduced detections to %zu", indices.size());
    jsi::Array detections(runtime, indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
      int idx = indices[i];
      jsi::Object detection(runtime);
      detection.setProperty(runtime, "class_id", jsi::Value(class_ids[idx]));
      detection.setProperty(runtime, "class_name", jsi::Value(runtime, jsi::String::createFromUtf8(runtime, getClassName(class_ids[idx], classes).c_str())));
      detection.setProperty(runtime, "confidence", jsi::Value(confidences[idx]));
      jsi::Array boxArray(runtime, 4);
      boxArray.setValueAtIndex(runtime, 0, jsi::Value(boxes[idx].x));
      boxArray.setValueAtIndex(runtime, 1, jsi::Value(boxes[idx].y));
      boxArray.setValueAtIndex(runtime, 2, jsi::Value(boxes[idx].width));
      boxArray.setValueAtIndex(runtime, 3, jsi::Value(boxes[idx].height));
      detection.setProperty(runtime, "box", boxArray);
      detections.setValueAtIndex(runtime, i, detection);
    }
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Returning %zu detections", indices.size());
    return detections;
  };

  auto func = jsi::Function::createFromHostFunction(runtime,
      jsi::PropNameID::forUtf8(runtime, "processOnnxFrame"),
      10,
      onnxProcessor);
  runtime.global().setProperty(runtime, "processOnnxFrame", func);
  __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "processOnnxFrame registered");
}
