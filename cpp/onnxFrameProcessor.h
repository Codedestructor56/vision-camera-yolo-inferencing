#ifndef ONNX_FRAME_PROCESSOR_H
#define ONNX_FRAME_PROCESSOR_H

#include <jsi/jsi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <android/log.h>

#ifdef HAVE_OPENCL
#include <opencv2/core/ocl.hpp>
#endif

#if defined(__ANDROID_API__) && (__ANDROID_API__ >= 29)
#include <android/NeuralNetworks.h>
#endif

using namespace facebook;
using namespace jsi;
inline std::string getClassName(int id, const std::vector<std::string>& classes) {
  return (id >= 0 && id < static_cast<int>(classes.size())) ? classes[id] : "unknown";
}

/*
 * OnnxFrameProcessor
 *
 * This class encapsulates all state related to the frame processor,
 * including the loaded model (as a cv::dnn::Net), the chosen device/backends,
 * and any acceleration settings. It is designed so that the model is loaded
 * only once (if the same model is used), and device checks (like OpenCL or NNAPI)
 * are only performed during the initial load.
 */
class OnnxFrameProcessor {
public:
  OnnxFrameProcessor();
  ~OnnxFrameProcessor();

  void loadModel(const std::string &modelPath, const std::string &modelType);

  std::vector<std::string> processFrame(const cv::Mat &image,
                                          const std::vector<std::string> &classes,
                                          float modelConfidenceThreshold,
                                          float modelScoreThreshold,
                                          float modelNMSThreshold);

  bool isModelLoaded() const { return modelLoaded; }

  static void registerOnnxFrameProcessor(Runtime &runtime);
private:
  cv::dnn::Net net;
  std::string currentModelPath;
  std::string currentModelType;
  bool modelLoaded;
  bool accelerationSet;

  void configureDevice(const std::string &modelType);

  void clearState();
  std::vector<std::string> runInference(const cv::Mat &blob,
                                          const std::vector<std::string> &classes,
                                          float modelConfidenceThreshold,
                                          float modelScoreThreshold,
                                          float modelNMSThreshold);
};

#endif
