#include <jni.h>
#include <jsi/jsi.h>
#include "react-native-vision-jsi-processor.h"

using namespace facebook;

extern "C"
JNIEXPORT void JNICALL
Java_com_visionjsiprocessoronnx_VisionJsiProcessorModule_nativeInstall(JNIEnv *env, jclass type, jlong jsi) {
    auto runtime = reinterpret_cast<jsi::Runtime *>(jsi);
    if (runtime) {
        visionjsiprocessor::install(*runtime);
    }
}
