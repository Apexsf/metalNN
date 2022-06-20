//
//  main.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#import <Foundation/Foundation.h>
#include <chrono>
#include <iostream>
#include <string>
#include "add.h"
#include "conv.h"

//int main(int argc, const char * argv[]) {
//    @autoreleasepool {
//        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
//        add op_add(resource);
//        for(int i = 0; i < 10000; i ++) {
//            auto start = std::chrono::steady_clock::now();
//            // insert code here...
//            op_add.execute();
//            float* res= op_add.getOutput();
//            auto end = std::chrono::steady_clock::now();
//            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//            std::cout  << duration << "  ";
//        }
//    }
//    return 0;
//}


int main() {
    CFBundleRef bundle = CFBundleGetMainBundle();
    CFURLRef url = CFBundleCopyBundleURL(bundle);
    CFStringRef string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
    CFRelease(url);
    const char* cString = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
    NSString* str = [NSString stringWithUTF8String:cString];
    CFRelease(string);
    

    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/out.bin";
    std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/weight.bin";
    std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bias.bin";

    std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
    tensor input_tensor(2,11,71,83);
    convParams convp{5,3,11,9,4,9,4,7};
    
    
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    id<MTLBuffer> inBuffer = [resource->getDevice() newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    
    shape outShape = conv::calOutShape(input_tensor.getShape(), convp);

    uint outMemSize = outShape.batch * roundUp(outShape.channel, 4) * outShape.width * outShape.height;
    uint outSize =outShape.batch * outShape.channel * outShape.width * outShape.height;
    

    id<MTLBuffer> outBuffer = [resource->getDevice() newBufferWithLength:outMemSize * sizeof(float) options:MTLResourceStorageModeShared];
    

    memcpy(inBuffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));

    
    float* torchOut =(float*) malloc(outMemSize*float(4));

    readDataFromFile(output_path.c_str(), outMemSize*sizeof(float), (void*)torchOut);
    
    tensor weight(convp.outC,convp.inC,convp.kernelH, convp.kernelW);
    weight.loadFromFile(weight_path.c_str());
    tensor bias(1, convp.outC, 1,1);
    bias.loadFromFile(bias_path.c_str());
    
    conv conv_op(resource, "conv", convp);
    conv_op.loadWeight(weight, &bias);
    float* outP = (float* ) outBuffer.contents;
    
    
    conv_op.execute(inBuffer, outBuffer, conv::makeConvConstant(input_tensor.getShape(), convp));
    tensor output_tensor(outShape);
    output_tensor.loadFromMemory(outP, tensor::interpOrder::NC4HW4);
    output_tensor.reInterpret(tensor::interpOrder::NCHW);
    outP = output_tensor.getRawPointer();
    
    double diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < outSize; ++i){
        diff +=std::abs( (outP[i] - torchOut[i]) );
        if (outP[i] != torchOut[i]){
            diff_cnt+=1;
        }
        if (diff > 1){
            std::cout;
        }
    }
    
    std::cout << "diff : " << diff << std::endl;
    free(torchOut);
    
    return 0;
    
    
    
}
