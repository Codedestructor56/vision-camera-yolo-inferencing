# react-native-vision-jsi-processor-onnx

A VisionCamera Frame Processor Plugin using JSI to run ONNX/TFLite models with OpenCV DNN on Android.

## Installation

```sh
npm install react-native-vision-jsi-processor-onnx
```

### Android

Ensure the following dependencies are added to your `android/app/build.gradle` file:

```groovy
android {
    packagingOptions {
        pickFirst 'META-INF/*'
    }
}
```

Also, ensure that OpenCV and other native dependencies are correctly integrated into your project.

## Usage

- Register the plugin in your native code to expose the JSI function `processOnnxFrame`.
- Use the install function to install it first.
- Use the `processOnnxFrame` inside a VisionCamera frame processor to run inference on camera frames.
- Supports both ONNX and TFLite models.



## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
