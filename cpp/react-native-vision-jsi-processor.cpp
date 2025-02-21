#include "react-native-vision-jsi-processor.h"
#include "onnxFrameProcessor.h"
#include <jsi/jsi.h>
#include "../android/react-native-vision-camera/android/src/main/cpp/frameprocessors/FrameHostObject.h"
#include <android/log.h>

namespace visionjsiprocessor {
  using namespace facebook;

  void install(jsi::Runtime& runtime) {
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: start");

      if (runtime.global().hasProperty(runtime, "frameProcessor")) {
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: already installed, skipping");
          return;
      }

      auto basePlugin = [=](jsi::Runtime& runtime,
                            const jsi::Value& thisArg,
                            const jsi::Value* args,
                            size_t count) -> jsi::Value {
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: start");
          auto valueAsObject = args[0].getObject(runtime);
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: obtained object from args[0]");
          auto frame = std::static_pointer_cast<vision::FrameHostObject>(valueAsObject.getHostObject(runtime));
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: obtained FrameHostObject");
          int frameHeight = 0;
#ifdef ANDROID
          frameHeight = frame->getFrame()->getHeight();
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: frameHeight (via getFrame()->getHeight()) = %d", frameHeight);
#else
          frameHeight = frame->frame.height;
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: frameHeight (via frame.height) = %d", frameHeight);
#endif
          __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "basePlugin: returning frameHeight");
          return jsi::Value(frameHeight);
      };
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: basePlugin lambda defined");

      auto jsiFunc = jsi::Function::createFromHostFunction(runtime,
          jsi::PropNameID::forUtf8(runtime, "frameProcessor"),
          1,
          basePlugin);
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: jsi function created");

      runtime.global().setProperty(runtime, "frameProcessor", jsiFunc);
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: frameProcessor set on global object");

      OnnxFrameProcessor::registerOnnxFrameProcessor(runtime);
      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: registerOnnxFrameProcessor called");

      __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "install: end");
  }
}
