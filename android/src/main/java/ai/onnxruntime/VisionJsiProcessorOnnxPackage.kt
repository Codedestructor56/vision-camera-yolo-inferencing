package ai.onnxruntime

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

// If your actual native module class is defined in a different package,
// you can reference it by its fully qualified name.
class VisionJsiProcessorOnnxPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        // If your module is defined as, for example, VisionJsiProcessorModule in com.visionjsiprocessoronnox,
        // you can instantiate it here:
        return listOf(com.visionjsiprocessoronnx.VisionJsiProcessorModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}
