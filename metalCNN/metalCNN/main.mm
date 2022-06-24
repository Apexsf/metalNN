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
#include "bn.h"

void test_conv(){
    
    CFBundleRef bundle = CFBundleGetMainBundle();
    CFURLRef url = CFBundleCopyBundleURL(bundle);
    CFStringRef string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
    CFRelease(url);
    const char* cString = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
    NSString* str = [NSString stringWithUTF8String:cString];
    CFRelease(string);
    

    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/out.bin";
    std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/weight.bin";
    std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/bias.bin";

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
    std::map<std::string, tensor> convWeights = {
        {"weight", std::move(weight)},
        {"bias", std::move(bias)}
    };
    conv_op.loadWeight(convWeights);
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
}

void test_bn() {
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/out.bin";
    std::string gamma_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/bn/gamma.bin";
    std::string beta_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/bn/beta.bin";
    std::string rm_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/bn/running_mean.bin";
    std::string rv_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/running_var.bin";
    std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
    shape shp{2,64,64,64};
    tensor input_tensor(shp);
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    
    tensor output_tensor(shp);
    output_tensor.loadFromFile(output_path.c_str());
    
    id<MTLBuffer> input_buffer = [resource->getDevice()
                                  newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [resource->getDevice() newBufferWithLength:output_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    
    memcpy(input_buffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));
    
    tensor gamma(1,shp.channel,1,1);
    tensor beta(1,shp.channel,1,1);
    tensor rm(1,shp.channel,1,1);
    tensor rv(1,shp.channel,1,1);
    gamma.loadFromFile(gamma_path.c_str());
    beta.loadFromFile(beta_path.c_str());
    rm.loadFromFile(rm_path.c_str());
    rv.loadFromFile(rv_path.c_str());
    
    std::map<std::string, tensor> bnWeights = {
        {"gamma", std::move(gamma)},
        {"beta", std::move(beta)},
        {"running_mean", std::move(rm)},
        {"running_var", std::move(rv)}
    };
    
    bn bn_op(resource, std::string("bn"), 64);
    bn_op.loadWeight(bnWeights);
    
    bn_op.execute(input_buffer, output_buffer, shp);
    
//    float* cmlout = (float*)output_buffer.contents;
    tensor cmlout_tensor(shp);
    cmlout_tensor.loadFromMemory((float*)output_buffer.contents, tensor::interpOrder::NC4HW4);
    
    cmlout_tensor.reInterpret(tensor::interpOrder::NCHW);
    float* cmlout_p = cmlout_tensor.getRawPointer();
    float* torchout_p = (float*)output_tensor.getRawPointer();
    
    uint outSize = cmlout_tensor.getShape().size();
    double diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < outSize; ++i){
        diff +=std::abs( (cmlout_p[i] - torchout_p[i]) );
        if (cmlout_p[i] != torchout_p[i]){
            diff_cnt+=1;
        }
        if (diff > 1){
            std::cout;
        }
    }
    std::cout << "diff : " << diff << std::endl;
//    -0.800417065
//    -0.976112008
//    -0.175694913
}
int main() {
//    test_conv();
    test_bn();
    
    
}
