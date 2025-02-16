package com.visionjsiprocessoronnx

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

        @JvmStatic external fun nativeInstall(runtimePtr: Long)
        @JvmStatic external fun nativeSetCallInvoker(callInvokerPtr: Long)
    }

    override fun getName(): String = NAME

    @ReactMethod(isBlockingSynchronousMethod = true)
    fun install() {
        val catalyst = reactApplicationContext.catalystInstance
        if (catalyst == null) {
            return
        }

        val jsContextHolderAny: Any? = catalyst.javaScriptContextHolder
        val runtimePtr: Long = if (jsContextHolderAny is JavaScriptContextHolder) {
            jsContextHolderAny.get()
        } else {
            try {
                val getMethod = jsContextHolderAny?.javaClass?.getMethod("get")
                getMethod?.invoke(jsContextHolderAny) as? Long ?: 0L
            } catch (e: Exception) {
                0L
            }
        }
        val jsCallInvokerHolderAny: Any? = catalyst.jsCallInvokerHolder
        val callInvokerPtr: Long = try {
            val getMethod = jsCallInvokerHolderAny?.javaClass?.getMethod("get")
            getMethod?.invoke(jsCallInvokerHolderAny) as? Long ?: 0L
        } catch (e: Exception) {
            0L
        }

        if (runtimePtr != 0L && callInvokerPtr != 0L) {
            nativeSetCallInvoker(callInvokerPtr)
            nativeInstall(runtimePtr)
        }
    }
}
