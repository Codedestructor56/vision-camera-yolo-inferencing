#include "../node_modules/react-native-vision-camera/android/src/main/cpp/frameprocessors/FrameHostObject.h"
#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#include <string>

using namespace facebook;
using namespace jsi;

extern "C" void registerFrameProcessorPlugins(Runtime &runtime) {
  auto modelInfoPlugin = [](Runtime &rt,
                            const Value &thisArg,
                            const Value *args,
                            size_t count) -> Value {
    if (count < 1) {
      throw JSError(rt, "Expected one argument");
    }
    if (!args[0].isString()) {
      throw JSError(rt, "Argument must be a string");
    }
    std::string path = args[0].asString(rt).utf8(rt);
    if (path.empty()) {
      throw JSError(rt, "Path is empty");
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInfo");
    Ort::SessionOptions session_options;
    try {
      Ort::Session session(env, path.c_str(), session_options);
      size_t inputCount = session.GetInputCount();
      size_t outputCount = session.GetOutputCount();
      Object result(rt);
      result.setProperty(rt, "inputCount", static_cast<double>(inputCount));
      result.setProperty(rt, "outputCount", static_cast<double>(outputCount));
      return result;
    } catch (const std::exception &e) {
      throw JSError(rt, e.what());
    }
  };

  auto jsiFunc = Function::createFromHostFunction(
      runtime,
      PropNameID::forUtf8(runtime, "getModelInfo"),
      1,
      modelInfoPlugin
  );
  runtime.global().setProperty(runtime, "getModelInfo", jsiFunc);

  if (!runtime.global().hasProperty(runtime, "getModelInfo")) {
    throw JSError(runtime, "Failed to set global property");
  }
}
