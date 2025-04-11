#include "onnxFrameProcessor.h"
#include "TypedArray.h"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace facebook;
using namespace jsi;

OnnxFrameProcessor::OnnxFrameProcessor()
    : modelLoaded(false) {
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Processor created (using ONNX Runtime via DCSP_CORE)");
}

OnnxFrameProcessor::~OnnxFrameProcessor() {
  clearState();
}

void OnnxFrameProcessor::clearState() {
  dcspCore.reset();
  currentModelPath.clear();
  currentModelType.clear();
  currentModelInputSize.clear();
  modelLoaded = false;
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "State cleared");
}

void OnnxFrameProcessor::loadModel(const std::string &modelPath, const std::string &modelType, int inputWidth, int inputHeight) {
  if (modelLoaded && currentModelPath == modelPath && currentModelType == modelType &&
      currentModelInputSize.size() == 2 && currentModelInputSize[0] == inputHeight && currentModelInputSize[1] == inputWidth) {
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "Model %s (%dx%d) already loaded", modelPath.c_str(), inputWidth, inputHeight);
    return;
  }

  clearState();

  currentModelPath = modelPath;
  currentModelType = modelType;
  currentModelInputSize = {inputHeight, inputWidth};

  __android_log_print(ANDROID_LOG_INFO, "OnnxFrameProcessor", "Loading ONNX model from: %s with input size %dx%d", modelPath.c_str(), inputWidth, inputHeight);

  if (modelType != "onnx") {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Invalid model type for DCSP_CORE integration: %s. Only 'onnx' is supported.", modelType.c_str());
      throw std::runtime_error("Invalid model type for DCSP_CORE, expected 'onnx'");
  }

  try {
    dcspCore = std::make_unique<DCSP_CORE>();

    DCSP_INIT_PARAM params;
    params.ModelPath = modelPath;
    params.ModelType = YOLO_ORIGIN_V8;
    params.imgSize = currentModelInputSize;

    params.RectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;

    params.CudaEnable = false;

    params.IntraOpNumThreads = 2;
    params.LogSeverityLevel = 3;

    char* createResult = dcspCore->CreateSession(params);
    if (createResult != RET_OK) {
        std::string errorMsg = "Failed to create ONNX Runtime session: ";
        errorMsg += createResult;
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "%s", errorMsg.c_str());
        dcspCore.reset();
        throw std::runtime_error(errorMsg);
    }

    modelLoaded = true;
    __android_log_print(ANDROID_LOG_INFO, "OnnxFrameProcessor", "ONNX Runtime session created successfully for %s", modelPath.c_str());

  } catch (const std::exception &e) {
    __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Exception during model loading: %s", e.what());
    clearState();
    throw;
  } catch (...) {
    __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Unknown exception during model loading");
    clearState();
    throw std::runtime_error("Unknown error during model loading");
  }
}

std::vector<std::string> OnnxFrameProcessor::processFrame(const cv::Mat &image,
                                                          const std::vector<std::string> &classes,
                                                          float modelConfidenceThreshold,
                                                          float modelNmsThreshold,
                                                          float modelScoreThreshold) {
    if (!modelLoaded || !dcspCore) {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Model not loaded, cannot process frame.");
        return {};
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", 
        "Processing frame with thresholds - Confidence: %.3f, NMS: %.3f",
        modelConfidenceThreshold, modelNmsThreshold);

    dcspCore->classes = classes;

    modelConfidenceThreshold = std::max(0.0f, std::min(1.0f, modelConfidenceThreshold));
    modelNmsThreshold = std::max(0.0f, std::min(1.0f, modelNmsThreshold));

    dcspCore->rectConfidenceThreshold = modelConfidenceThreshold;
    dcspCore->iouThreshold = modelNmsThreshold;

    std::vector<DCSP_RESULT> results;
    results.reserve(20);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat mutableImage = image.clone();
    char* runResult = dcspCore->RunSession(mutableImage, results);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", 
        "RunSession completed in %.2f ms, found %zu results.", 
        duration.count(), results.size());

    if (runResult != RET_OK) {
        std::string errorMsg = "Error during ONNX Runtime RunSession: ";
        errorMsg += runResult;
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "%s", errorMsg.c_str());
        return {};
    }

    std::vector<std::string> detections;
    detections.reserve(results.size());

    for (const auto& res : results) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(5);
        oss << "{ \"class_id\": " << res.classId
            << ", \"class_name\": \"" << getClassName(res.classId, classes) << "\""
            << ", \"confidence\": " << res.confidence
            << ", \"box\": [" << res.box.x << ", " << res.box.y
            << ", " << res.box.width << ", " << res.box.height << "] }";
        detections.push_back(oss.str());
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", 
        "Processing complete. Returning %zu detections", detections.size());

    return detections;
}

static std::shared_ptr<OnnxFrameProcessor> gProcessor = std::make_shared<OnnxFrameProcessor>();

void OnnxFrameProcessor::registerOnnxFrameProcessor(jsi::Runtime &runtime) {
  auto onnxProcessorFunc = [=](jsi::Runtime &runtime,
                               const jsi::Value &thisArg,
                               const jsi::Value *args,
                               size_t count) -> jsi::Value {
    auto start_time = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", "JSI processOnnxFrame called");

    const int expectedArgCount = 12;
    if (count != expectedArgCount) {
      __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Expected %d arguments, received %zu", expectedArgCount, count);
      throw jsi::JSError(runtime, "Expected " + std::to_string(expectedArgCount) + " arguments");
    }

    double rows = args[0].asNumber();
    double cols = args[1].asNumber();
    double channels = args[2].asNumber();
    jsi::Object input = args[3].asObject(runtime);
    
    auto inputBuffer = mrousavy::getTypedArray(runtime, std::move(input));
    auto kind = inputBuffer.getKind(runtime);
    bool isFloat32 = (kind == mrousavy::TypedArrayKind::Float32Array);
    
    int type = -1;
    if (channels == 1) { 
        type = isFloat32 ? CV_32F : CV_8U;
    } else if (channels == 3) {
        type = isFloat32 ? CV_32FC3 : CV_8UC3;
    } else if (channels == 4) {
        type = isFloat32 ? CV_32FC4 : CV_8UC4;
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Invalid channel count: %f", channels);
        throw jsi::JSError(runtime, "Invalid channel count passed to frameBufferToMat!");
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OnnxFrameProcessor", 
        "Image dims: %.0fx%.0f, channels: %.0f, data type: %s", 
        rows, cols, channels, isFloat32 ? "float32" : "uint8");

    auto vec = inputBuffer.toVector(runtime);
    cv::Mat image(static_cast<int>(rows), static_cast<int>(cols), type);
    
    size_t expectedSize = image.total() * image.elemSize();
    if (vec.size() != expectedSize) {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", 
            "TypedArray size (%zu) does not match image buffer size (%zu) for %s data", 
            vec.size(), expectedSize, isFloat32 ? "float32" : "uint8");
        throw jsi::JSError(runtime, "TypedArray size mismatch");
    }

    memcpy(image.data, vec.data(), vec.size());

    cv::Mat processImage;
    if (isFloat32) {
        image.convertTo(processImage, CV_8UC3, 255.0);
    } else {
        processImage = image;
    }

    if (processImage.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Image is empty after conversion");
        throw jsi::JSError(runtime, "Empty image");
    }

    std::string modelPath = args[4].asString(runtime).utf8(runtime);
    float modelConfidenceThreshold = static_cast<float>(args[5].asNumber());
    float modelNmsThreshold = static_cast<float>(args[6].asNumber());
    float modelScoreThreshold = static_cast<float>(args[7].asNumber());

    jsi::Array classNamesArray = args[8].asObject(runtime).asArray(runtime);
    size_t arrayLength = classNamesArray.length(runtime);
    if (arrayLength == 0) throw jsi::JSError(runtime, "Class names array cannot be empty");
    std::vector<std::string> classes;
    classes.reserve(arrayLength);
    for (size_t i = 0; i < arrayLength; i++) {
      jsi::Value val = classNamesArray.getValueAtIndex(runtime, i);
      if (!val.isString()) throw jsi::JSError(runtime, "Class names array must contain only strings");
      classes.push_back(val.asString(runtime).utf8(runtime));
    }

    std::string modelType = args[9].asString(runtime).utf8(runtime);
    int inputWidth = static_cast<int>(args[10].asNumber());
    int inputHeight = static_cast<int>(args[11].asNumber());

    if (inputWidth <= 0 || inputHeight <= 0) {
         throw jsi::JSError(runtime, "Invalid model input dimensions provided");
    }

    try {
        gProcessor->loadModel(modelPath, modelType, inputWidth, inputHeight);

        auto detections = gProcessor->processFrame(processImage, classes,
                                                   modelConfidenceThreshold,
                                                   modelNmsThreshold,
                                                   modelScoreThreshold);

        jsi::Array jsiDetections(runtime, detections.size());
        for (size_t i = 0; i < detections.size(); i++) {
          jsiDetections.setValueAtIndex(runtime, i,
              jsi::Value(runtime, jsi::String::createFromUtf8(runtime, detections[i])));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
        __android_log_print(ANDROID_LOG_INFO, "OnnxFrameProcessor", "JSI processOnnxFrame finished in %.2f ms, returning %zu detections", total_duration.count(), detections.size());
        return jsiDetections;

    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Exception in JSI call: %s", e.what());
        throw jsi::JSError(runtime, std::string("ONNX Processing Error: ") + e.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "OnnxFrameProcessor", "Unknown exception in JSI call");
        throw jsi::JSError(runtime, "Unknown ONNX Processing Error");
    }
  };

  auto func = jsi::Function::createFromHostFunction(runtime,
                 jsi::PropNameID::forUtf8(runtime, "processOnnxFrame"),
                 12,
                 onnxProcessorFunc);
  runtime.global().setProperty(runtime, "processOnnxFrame", func);
  __android_log_print(ANDROID_LOG_INFO, "OnnxFrameProcessor", "JSI function 'processOnnxFrame' (ONNX Runtime backend) registered");
}