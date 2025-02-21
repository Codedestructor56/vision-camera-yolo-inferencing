package com.visionjsiprocessoronnx

import android.util.Log
import com.facebook.react.bridge.JavaScriptContextHolder
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.module.annotations.ReactModule

@ReactModule(name = VisionJsiProcessorOnnxModule.NAME)
class VisionJsiProcessorOnnxModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "VisionJsiProcessor"
        init {
            System.loadLibrary("react-native-vision-jsi-processor-onnx")
        }
        @JvmStatic external fun nativeInstall(jsi: Long)
    }

    override fun getName(): String {
        return NAME
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    fun install() {
        val jsContext: JavaScriptContextHolder = reactApplicationContext.javaScriptContextHolder!!
        if (jsContext.get() != 0L) {
            nativeInstall(jsContext.get())
        } else {
            Log.e("VisionJSIProcessor", "JSI Runtime is not available in debug mode")
        }
    }
}


