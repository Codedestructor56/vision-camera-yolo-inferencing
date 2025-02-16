#include <jni.h>
#include <jsi/jsi.h>
#include <android/log.h>
#include <thread>
#include <sstream>
#include <ReactCommon/CallInvoker.h>
#include "react-native-vision-jsi-processor.h"

using namespace facebook;

std::string getThreadId() {
    std::ostringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

static std::shared_ptr<react::CallInvoker> jsCallInvoker;

extern "C"
JNIEXPORT void JNICALL
Java_com_visionjsiprocessoronnx_VisionJsiProcessorModule_nativeInstall(JNIEnv *env, jclass clazz, jlong runtimePtr) {
    auto *runtime = reinterpret_cast<jsi::Runtime *>(runtimePtr);

    if (!runtime) {
        __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "JSI Runtime is null!");
        return;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "Runtime pointer: %p", (void *)runtime, getThreadId().c_str());

    if (!jsCallInvoker) {
        __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "CallInvoker is not set!");
        return;
    }

    jsCallInvoker->invokeAsync([runtime]() {
        try {
            visionjsiprocessor::install(*runtime);
            __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "VisionJSIProcessor installed successfully on JS thread.");
        } catch (const std::exception &e) {
            __android_log_print(ANDROID_LOG_ERROR, "VisionJSIProcessor", "Exception during install: %s", e.what());
        }
    });
}

extern "C"
JNIEXPORT void JNICALL
Java_com_visionjsiprocessoronnx_VisionJsiProcessorModule_nativeSetCallInvoker(JNIEnv *env, jclass clazz, jlong callInvokerPtr) {
    jsCallInvoker = *reinterpret_cast<std::shared_ptr<react::CallInvoker> *>(callInvokerPtr);
    __android_log_print(ANDROID_LOG_DEBUG, "VisionJSIProcessor", "CallInvoker set successfully.");
}
