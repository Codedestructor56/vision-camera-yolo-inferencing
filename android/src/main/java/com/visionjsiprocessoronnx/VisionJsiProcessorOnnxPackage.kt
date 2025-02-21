package com.visionjsiprocessoronnx

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class VisionJsiProcessorOnnxPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        // If your module is defined as, for example, VisionJsiProcessorModule in com.visionjsiprocessoronnox,
        // you can instantiate it here:
        return listOf(VisionJsiProcessorModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}


