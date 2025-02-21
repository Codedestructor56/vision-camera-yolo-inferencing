set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
cmake_minimum_required(VERSION 3.4.1)
project(VisionJSIProcessorONNX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)

find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS OpenCV::opencv_java4)


file(GLOB FRAMEPROCESSOR_SOURCES "react-native-vision-camera/android/src/main/cpp/frameprocessors/*.cpp")
file(GLOB JSIH_SOURCES "jsi/*.h")
file(GLOB JSICPP_SOURCES "jsi/*.cpp")
list(REMOVE_ITEM FRAMEPROCESSOR_SOURCES "react-native-vision-camera/android/src/main/cpp/frameprocessors/ModelInfoPlugin.cpp")
list(REMOVE_ITEM FASTOPENCVCPP_SOURCES "react-native-fast-opencv/cpp/onnxFrameProcessor.cpp")

add_library(react-native-vision-jsi-processor-onnx SHARED
    ../node_modules/react-native/ReactCommon/jsi/jsi/jsi.cpp
    ../cpp/react-native-vision-jsi-processor.cpp

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

target_compile_definitions(react-native-vision-jsi-processor-onnx PUBLIC HAVE_OPENCL)

target_link_libraries(
        react-native-vision-jsi-processor-onnx
        android
        ReactAndroid::jsi
        fbjni::fbjni
        VisionCamera
        log
        OpenCV::opencv_java4
)

if(ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 29)
    target_link_libraries(
            react-native-vision-jsi-processor-onnx
            android
            ReactAndroid::jsi
            fbjni::fbjni
            VisionCamera
            log
            OpenCV::opencv_java4
            neuralnetworks
    )
else()
    target_link_libraries(
            react-native-vision-jsi-processor-onnx
            android
            ReactAndroid::jsi
            fbjni::fbjni
            VisionCamera
            log
            OpenCV::opencv_java4
    )
endif()

if(ReactAndroid_VERSION_MINOR GREATER_EQUAL 76)
    target_link_libraries(
        ${PACKAGE_NAME}
        ReactAndroid::reactnative
    )
else()
    target_link_libraries(
        ${PACKAGE_NAME}
        ReactAndroid::reactnativejni
    )
endif()

