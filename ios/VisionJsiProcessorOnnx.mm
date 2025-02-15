#import "VisionJsiProcessorOnnx.h"

@implementation VisionJsiProcessorOnnx
RCT_EXPORT_MODULE()

- (NSNumber *)multiply:(double)a b:(double)b {
    NSNumber *result = @(visionjsiprocessoronnx::multiply(a, b));

    return result;
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeVisionJsiProcessorOnnxSpecJSI>(params);
}

@end
