#include "react-native-vision-jsi-processor.h"
#include <jsi/jsi.h>
#ifdef ANDROID
#include "../android/react-native-vision-camera/android/src/main/cpp/frameprocessors/FrameHostObject.h"
#else
#include "../../../node_modules/react-native-vision-camera/ios/Frame Processor/FrameHostObject.h"
#endif

extern "C" void registerOnnxFrameProcessor(facebook::jsi::Runtime &runtime);

namespace visionjsiprocessor {
  using namespace facebook;

  void install(jsi::Runtime& runtime) {
      auto basePlugin = [=](jsi::Runtime& runtime,
                            const jsi::Value& thisArg,
                            const jsi::Value* args,
                            size_t count) -> jsi::Value {
          auto valueAsObject = args[0].getObject(runtime);

          auto frame = std::static_pointer_cast<vision::FrameHostObject>(valueAsObject.getHostObject(runtime));
          int frameHeight = 0;
#ifdef ANDROID
          frameHeight = frame->getFrame()->getHeight();
#else
          frameHeight = frame->frame.height;
#endif
          return jsi::Value(frameHeight);
      };

      auto jsiFunc = jsi::Function::createFromHostFunction(runtime,
          jsi::PropNameID::forUtf8(runtime, "frameProcessor"),
          1,
          basePlugin);
      runtime.global().setProperty(runtime, "frameProcessor", jsiFunc);

      registerOnnxFrameProcessor(runtime);
  }
}
