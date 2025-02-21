#include <jni.h>
#include <jsi/jsi.h>
#include <android/log.h>
#include <thread>
#include <sstream>
#include <mutex>
#include <chrono>
#include "react-native-vision-jsi-processor.h"

using namespace facebook;


std::string getThreadId() {
    std::ostringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

static std::timed_mutex installMutex;

extern "C"
JNIEXPORT void JNICALL
Java_com_visionjsiprocessoronnx_VisionJsiProcessorModule_nativeInstall(JNIEnv *env, jclass clazz, jlong runtimePtr) {
    auto *runtime = reinterpret_cast<jsi::Runtime *>(runtimePtr);
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor",
                          "Runtime pointer: %p, Thread: %s",
                          (void *)runtime, getThreadId().c_str());

    if (!runtime) {
        __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "JSI Runtime is null!");
        return;
    }

    installMutex.lock();
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Acquired mutex, calling install");
    visionjsiprocessor::install(*runtime);
    installMutex.unlock();
}
