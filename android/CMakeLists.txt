cmake_minimum_required(VERSION 3.4.1)
project(VisionJSIProcessorONNX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)

find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS OpenCV::opencv_java4)
#find_package(react-native-vision-camera REQUIRED CONFIG)


file(GLOB FRAMEPROCESSOR_SOURCES "react-native-vision-camera/android/src/main/cpp/frameprocessors/*.cpp")
file(GLOB JSIH_SOURCES "jsi/*.h")
file(GLOB JSICPP_SOURCES "jsi/*.cpp")
list(REMOVE_ITEM FRAMEPROCESSOR_SOURCES "react-native-vision-camera/android/src/main/cpp/frameprocessors/ModelInfoPlugin.cpp")
list(REMOVE_ITEM FASTOPENCVCPP_SOURCES "react-native-fast-opencv/cpp/onnxFrameProcessor.cpp")

add_library(react-native-vision-jsi-processor-onnx SHARED
    ../node_modules/react-native/ReactCommon/jsi/jsi/jsi.cpp
    ../cpp/react-native-vision-jsi-processor.cpp
    #../cpp/onnxFrameProcessorPlugin.cpp
    ../cpp/onnxFrameProcessor.cpp
    ../cpp/cpp-addapter.cpp
    ${FRAMEPROCESSOR_SOURCES}
    ${JSIH_SOURCES}
    ${JSICPP_SOURCES}
)

target_include_directories(react-native-vision-jsi-processor-onnx PUBLIC
    "../node_modules/react-native/ReactAndroid/src/main/jni/react/turbomodule/ReactCommon"
    "../node_modules/react-native/ReactCommon/callinvoker"
    "react-native-vision-camera/android/src/main/cpp/frameprocessors"
    "react-native-vision-camera/android/src/main/cpp"
    "jsi"
)

include_directories(
    ../node_modules/react-native/React
    ../node_modules/react-native/React/Base
    ../node_modules/react-native/ReactCommon/jsi
    react-native-vision-camera/android/src/main/cpp/frameprocessors
    react-native-vision-camera/android/src/main/cpp/frameprocessors/java-bindings
)

add_library(VisionCamera SHARED IMPORTED)
set_target_properties(VisionCamera PROPERTIES IMPORTED_LOCATION
  "${CMAKE_CURRENT_SOURCE_DIR}/react-native-vision-camera/android/build/intermediates/merged_native_libs/debug/mergeDebugNativeLibs/out/lib/${ANDROID_ABI}/libVisionCamera.so"
)

target_link_libraries(
        react-native-vision-jsi-processor-onnx
        android
        ReactAndroid::jsi
        fbjni::fbjni
        VisionCamera
        log
        OpenCV::opencv_java4
)

