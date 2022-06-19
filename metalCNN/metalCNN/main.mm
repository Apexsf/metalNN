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
    
//    std::string input_path =std::string(cString) + "/resource/input.bin";
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/out.bin";
    std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/weight.bin";
    
    std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
    convParams convp{3,3,4,8,1,1,2,2};
    conv conv_op(resource, "conv", convp);
    

    uint out_size = 2 * 8 * 32 * 32;

    id<MTLBuffer> outBuffer = [resource->getDevice() newBufferWithLength:out_size * sizeof(float) options:MTLResourceStorageModeShared];
    
    tensor input_tensor(2,4,64,64);
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    
    id<MTLBuffer> inBuffer = [resource->getDevice() newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];

    memcpy(inBuffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));

    
    float* out_p =(float*) malloc(out_size*float(4));

    readDataFromFile(output_path.c_str(), out_size*sizeof(float), (void*)out_p);
    
    tensor weight(8,4,3,3);
    weight.loadFromFile(weight_path.c_str());
    conv_op.loadWeight(weight);
    
    float* in_p = (float*) inBuffer.contents;
    float* out_c_p = (float* ) outBuffer.contents;
    

    
    convRunTimeConstant crtc{2,1,64*64,64,64,
        2,2,32*32,32,32,
        3,3,9,2,2,1,1
    };
    
    conv_op.execute(inBuffer, outBuffer, crtc);
    
    tensor output_tensor(2,8,32,32);
    output_tensor.loadFromMemory(out_c_p, tensor::interpOrder::NC4HW4);
    output_tensor.reInterpret(tensor::interpOrder::NCHW);
    out_c_p = output_tensor.getRawPointer();
    
    double diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < out_size; ++i){
        diff +=std::abs( (out_c_p[i] - out_p[i]) );
        if (out_c_p[i] != out_p[i]){
            diff_cnt+=1;
        }
    }
    
    return 0;
    
    
    
}
