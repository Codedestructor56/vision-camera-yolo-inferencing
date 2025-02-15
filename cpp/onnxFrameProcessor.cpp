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
    if (count < 10) throw jsi::JSError(runtime, "Expected 10 arguments");
    auto rows = args[0].asNumber();
    auto cols = args[1].asNumber();
    auto channels = args[2].asNumber();
    auto input = args[3].asObject(runtime);
    int type = -1;
    if (channels == 1) {
      type = CV_8U;
    } else if (channels == 3) {
      type = CV_8UC3;
    } else if (channels == 4) {
      type = CV_8UC4;
    } else {
      throw std::runtime_error("Invalid channel count passed to frameBufferToMat!");
    }
    auto inputBuffer = mrousavy::getTypedArray(runtime, std::move(input));
    auto vec = inputBuffer.toVector(runtime);
    cv::Mat image(rows, cols, type);
    memcpy(image.data, vec.data(), (int)rows * (int)cols * (int)channels);
    if (image.empty()) throw jsi::JSError(runtime, "Empty image");
    std::string modelPath = args[4].asString(runtime).utf8(runtime);
    std::string modelType = args[9].asString(runtime).utf8(runtime);
    cv::dnn::Net net;
    if (modelType == "onnx") {
      net = cv::dnn::readNetFromONNX(modelPath);
      if (net.empty()) throw jsi::JSError(runtime, "Failed to load ONNX model");
    } else if (modelType == "tflite") {
      net = cv::dnn::readNetFromTFLite(modelPath);
      if (net.empty()) throw jsi::JSError(runtime, "Failed to load TFLite model");
    } else {
      throw std::runtime_error("Invalid model type, expected 'tflite' or 'onnx'");
    }
    int origHeight = image.rows;
    int origWidth = image.cols;
    int length = std::max(origWidth, origHeight);
    cv::Mat squareImage(length, length, image.type(), cv::Scalar(0, 0, 0));
    image.copyTo(squareImage(cv::Rect(0, 0, origWidth, origHeight)));
    float scaley = static_cast<float>(origHeight) / static_cast<float>(length);
    float scalex = static_cast<float>(origWidth) / static_cast<float>(length);
    cv::Mat blob;
    cv::dnn::blobFromImage(squareImage, blob, 1/255.0, cv::Size(length, length), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    int rows_out = outputs[0].size[1];
    int dimensions = outputs[0].size[2];
    bool yolov8 = true;
    if (dimensions > rows_out) {
      rows_out = outputs[0].size[2];
      dimensions = outputs[0].size[1];
      outputs[0] = outputs[0].reshape(1, dimensions);
      cv::transpose(outputs[0], outputs[0]);
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
          int left = static_cast<int>((x - 0.5 * w) * scalex);
          int top = static_cast<int>((y - 0.5 * h) * scaley);
          int boxW = static_cast<int>(w * scalex);
          int boxH = static_cast<int>(h * scaley);
          boxes.push_back(cv::Rect(left, top, boxW, boxH));
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
            int left = static_cast<int>((x - 0.5 * w) * scalex);
            int top = static_cast<int>((y - 0.5 * h) * scaley);
            int right = static_cast<int>((x + 0.5 * w) * scalex);
            int bottom = static_cast<int>((y + 0.5 * h) * scaley);
            boxes.push_back(cv::Rect(left, top, right, bottom));
          }
        }
      }
      data += dimensions;
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indices);
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
      detection.setProperty(runtime, "scalex", jsi::Value(scalex));
      detection.setProperty(runtime, "scaley", jsi::Value(scaley));
      detections.setValueAtIndex(runtime, i, detection);
    }
    return detections;
  };

  auto func = jsi::Function::createFromHostFunction(runtime,
      jsi::PropNameID::forUtf8(runtime, "processOnnxFrame"),
      10,
      onnxProcessor);
  runtime.global().setProperty(runtime, "processOnnxFrame", func);
}
