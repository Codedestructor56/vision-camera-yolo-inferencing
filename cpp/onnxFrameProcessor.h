#ifndef ONNX_FRAME_PROCESSOR_H
#define ONNX_FRAME_PROCESSOR_H

#include <jsi/jsi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <android/log.h>
#include <memory>

#include "Inference.h"

using namespace facebook;
using namespace jsi;

inline std::string getClassName(int id, const std::vector<std::string>& classes) {
  return (id >= 0 && id < static_cast<int>(classes.size())) ? classes[id] : "unknown";
}

class OnnxFrameProcessor {
public:
  OnnxFrameProcessor();
  ~OnnxFrameProcessor();

  void loadModel(const std::string &modelPath, const std::string &modelType, int inputWidth, int inputHeight);

  std::vector<std::string> processFrame(const cv::Mat &image,
                                          const std::vector<std::string> &classes,
                                          float modelConfidenceThreshold,
                                          float modelNmsThreshold,
                                          float modelScoreThreshold);

  bool isModelLoaded() const { return modelLoaded; }

  static void registerOnnxFrameProcessor(Runtime &runtime);
private:
  std::unique_ptr<DCSP_CORE> dcspCore;
  std::string currentModelPath;
  std::string currentModelType;
  std::vector<int> currentModelInputSize;
  bool modelLoaded;

  void clearState();
};

#endif