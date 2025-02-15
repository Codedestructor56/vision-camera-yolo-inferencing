# Custom config file for react-native-vision-camera

if(NOT TARGET react-native-vision-camera::VisionCamera)
  add_library(VisionCamera SHARED IMPORTED)
  set_target_properties(VisionCamera PROPERTIES
    # Point to the prebuilt libVisionCamera.so in the local build folder.
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../build/intermediates/merged_native_libs/debug/out/lib/${ANDROID_ABI}/libVisionCamera.so"
    # Use the headers provided in the src/main/cpp directory.
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../src/main/cpp"
  )
  # Create an alias so that the target can be referenced as react-native-vision-camera::VisionCamera.
  add_library(react-native-vision-camera::VisionCamera ALIAS VisionCamera)
endif()
