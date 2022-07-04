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
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "pooling.h"
#include "elemWise.h"
#include "convBnRelu.h"
#include "matmul.h"
#include "resnet.h"
#include "builder.h"


std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();

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
    
    std::vector<id<MTLBuffer>> inOutBuffers{inBuffer, outBuffer};
    convConstant convConst = conv::makeConvConstant(input_tensor.getShape(), convp);
    conv_op.runOnce(inOutBuffers, &convConst);
    
    
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
    shape shp{2,64,67,67};
    tensor input_tensor(shp);
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    
    tensor output_tensor(shp);
    output_tensor.loadFromFile(output_path.c_str());
    
    id<MTLBuffer> input_buffer = [resource->getDevice()
                                  newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [resource->getDevice() newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    
    memcpy(input_buffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));
    
//    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/basicBlock/testData.json";
//    NSData *data = [NSData dataWithContentsOfFile:path];
//    NSError *error;
//
//    auto jsonObject = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
//    auto gammaW = jsonObject[@"weights"][@"gamma"];
//    auto betaW = jsonObject[@"weights"][@"beta"];
//    auto rmW = jsonObject[@"weights"][@"running_mean"];
//    auto rvW = jsonObject[@"weights"][@"running_var"];
//
//    float gammaM[64];
//    float betaM[64];
//    float rmM[64];
//    float rvM[64];
//
//    for(uint i = 0; i < 64; ++i){
//        gammaM[i] = [[gammaW objectAtIndex:i]floatValue];
//        betaM[i] = [[betaW objectAtIndex:i]floatValue];
//        rmM[i] = [[rmW objectAtIndex:i]floatValue];
//        rvM[i] = [[rvW objectAtIndex:i]floatValue];
//    }
//
    tensor gamma(1,shp.channel,1,1);
    tensor beta(1,shp.channel,1,1);
    tensor rm(1,shp.channel,1,1);
    tensor rv(1,shp.channel,1,1);
//    gamma.loadFromMemory(gammaM, tensor::interpOrder::NCHW);
//    beta.loadFromMemory(betaM, tensor::interpOrder::NCHW);
//    rm.loadFromMemory(rmM, tensor::interpOrder::NCHW);
//    rv.loadFromMemory(rvM, tensor::interpOrder::NCHW);
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
    std::vector<id<MTLBuffer>> inOutBuffers{input_buffer, output_buffer};
    bnConstant bnC{2,(int)divUp(shp.channel, 4), (int)shp.height, (int)shp.width, (int)(shp.height * shp.width)};
    bn_op.runOnce(inOutBuffers, &bnC);
    
    tensor cmlout_tensor(shp);
    cmlout_tensor.loadFromMemory((float*)output_buffer.contents, tensor::interpOrder::NC4HW4);
    
    cmlout_tensor.reInterpret(tensor::interpOrder::NCHW);
    float* cmlout_p = cmlout_tensor.getRawPointer();
    float* torchout_p = (float*)output_tensor.getRawPointer();
    
    uint outSize = cmlout_tensor.getShape().size();
    float diff;
    float total_diff = 0;
    float max_diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < outSize; ++i){
        diff = std::abs( (cmlout_p[i] - torchout_p[i]) );
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        if (cmlout_p[i] != torchout_p[i]){
            diff_cnt+=1;
        }
        if (total_diff > 1){
            std::cout;
        }
    }
    std::cout << "total diff : " << total_diff << std::endl;
    std::cout << "max diff : " << max_diff << std::endl;
}


id<MTLBuffer> makingInputBuffer(std::string input_path, const shape& shp){
    tensor input_tensor(shp);
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    id<MTLBuffer> input_buffer = [resource->getDevice()
                                  newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    memcpy(input_buffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));
    return input_buffer;
}

tensor makingTorchOutTensorNCHW(std::string output_path, const shape& shp) {
    tensor torchOutTensor(shp);
    torchOutTensor.loadFromFile(output_path.c_str());
    return torchOutTensor;
}

tensor makingMetalOutTensorNCHW(id<MTLBuffer> outBuffer, const shape& shp){
    tensor metalOutTensor(shp);
    metalOutTensor.loadFromMemory((float*)outBuffer.contents, tensor::interpOrder::NC4HW4   );
    metalOutTensor.reInterpret(tensor::interpOrder::NCHW);
    return metalOutTensor;
}

void diffProfile(float* p1, float* p2, size_t size){

    float diff;
    float total_diff = 0;
    float max_diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < size; ++i){
        diff = std::abs( (p1[i] - p2[i]) );
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        if (p1[i] != p2[i]){
            diff_cnt+=1;
        }
        if (total_diff > 1){
            std::cout;
        }
    }
    std::cout << "total diff : " << total_diff << std::endl;
    std::cout << "max diff : " << max_diff << std::endl;
}



void test_act() {
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/act/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/act/out.bin";
    shape shp{2,11,71,83};
    id<MTLBuffer>  inputBuffer = makingInputBuffer(input_path, shp);
    id<MTLBuffer> outputBuffer = [resource->getDevice() newBufferWithLength:inputBuffer.length options:MTLResourceStorageModeShared];
    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, shp);
    
    act reluOp(resource, std::string("relu"));
    std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
    actConstant reluConst{(int)shp.batch, (int)divUp(shp.channel, 4), (int)shp.height, (int)shp.width};
    reluOp.runOnce(inOutBuffers, &reluConst);
    
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, shp);
    
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}


void test_pooling() {
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/pooling/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/pooling/out.bin";
    shape inShape{2,11,71,83};
    poolingParams poolingPara {9,7,4,3,4,7};
//    shape inShape{2,7,16,16};
//    poolingParams poolingPara{9,7,2,3,2,2};
    poolingConstant poolingConst = pooling::makePoolingConstant(inShape, poolingPara);
    shape outShape {(uint)poolingConst.out_batch, inShape.channel, (uint)poolingConst.out_height, (uint)poolingConst.out_width};
    
    id<MTLBuffer>  inputBuffer = makingInputBuffer(input_path, inShape);
    id<MTLBuffer> outputBuffer = [resource->getDevice() newBufferWithLength:poolingConst.out_batch * poolingConst.out_slice * 4 * poolingConst.out_height * poolingConst.out_height * sizeof(float) options:MTLResourceStorageModeShared];
    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    
    pooling poolingOp(resource, std::string("poolingAvg"), poolingPara);
    std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
    
    for(int i = 0; i < 1000000; i++){
        poolingOp.runOnce(inOutBuffers, &poolingConst);
    }
   
    
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}

void test_elemWise() {
    std::string input1_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/input1.bin";
    std::string input2_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/input2.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/out.bin";
    shape shp{2,11,71,83};
    id<MTLBuffer>  inputBuffer1 = makingInputBuffer(input1_path, shp);
    id<MTLBuffer>  inputBuffer2 = makingInputBuffer(input2_path, shp);
    id<MTLBuffer> outputBuffer = [resource->getDevice() newBufferWithLength:inputBuffer1.length options:MTLResourceStorageModeShared];
    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, shp);
    
    elemWise elemWiseOp(resource, std::string("elemWiseMul"));
    std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer1, inputBuffer2, outputBuffer};
    elemWiseConstant elemWiseConst{(int)shp.batch, (int)divUp(shp.channel, 4), (int)shp.height, (int)shp.width, (int)shp.width * (int)shp.height};
    
    elemWiseOp.runOnce(inOutBuffers, &elemWiseConst);
    
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, shp);
    
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}

void test_convBnRelu() {
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/out.bin";
    std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/weight.bin";
    std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/bias.bin";
    std::string gamma_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/gamma.bin";
    std::string beta_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/beta.bin";
    std::string rm_path =
        "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/running_mean.bin";
    std::string rv_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBnRelu/running_var.bin";
    shape inShape {2,11,71,83};
    convParams convPara{5,3,11,9,4,9,4,7};
    convConstant convConst = conv::makeConvConstant(inShape, convPara);
    bnConstant bnConst{convConst.out_batch, convConst.out_slice, convConst.out_height, convConst.out_width, convConst.out_size};
    actConstant actConst{convConst.out_batch, convConst.out_slice, convConst.out_height, convConst.out_width};

    id<MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id<MTLBuffer> outputBuffer1 = [resource->getDevice() newBufferWithLength:convConst.out_batch * convConst.out_slice * 4 * convConst.out_height * convConst.out_height * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer2 = [resource->getDevice() newBufferWithLength:convConst.out_batch * convConst.out_slice * 4 * convConst.out_height * convConst.out_height * sizeof(float) options:MTLResourceStorageModeShared];
    
    convBnRelu CBR_OP(resource, convPara, convPara.outC);

    tensor weight(convPara.outC,convPara.inC,convPara.kernelH, convPara.kernelW);
    weight.loadFromFile(weight_path.c_str());
    tensor bias(1, convPara.outC, 1,1);
    bias.loadFromFile(bias_path.c_str());
    std::map<std::string, tensor> convWeights = {
        {"weight", std::move(weight)},
        {"bias", std::move(bias)}
    };
    CBR_OP.conv_.loadWeight(convWeights);
    
    
    uint bnChannel = convPara.outC;
    tensor gamma(1,bnChannel,1,1);
    tensor beta(1,bnChannel,1,1);
    tensor rm(1,bnChannel,1,1);
    tensor rv(1,bnChannel,1,1);
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
    
    CBR_OP.bn_.loadWeight(bnWeights);
    
    CBR_OP.run(inputBuffer, outputBuffer1, outputBuffer2, convConst, bnConst, actConst);
    
    shape OutShape{(uint)convConst.out_batch, convPara.outC, (uint)convConst.out_height, (uint)convConst.out_width};
    
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer1, OutShape);
    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, OutShape);
    
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
    
}



void testBufferPool() {
//    bufferPool pool (resource);
//    id <MTLBuffer> buffer = pool.get(1000);
//    pool.put(buffer.length / sizeof(float), buffer);
    id<MTLBuffer> buffer = resource->getBuffer(1002);
    resource->putBuffer(buffer.length / sizeof(float), buffer);
    std::cout;
}

void testBasicBlock(){
    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/basicBlock/testData.json";
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *error;
    NSDictionary *basicBlockInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
    
    basicBlock block = makingBasicBlock(resource, basicBlockInfo);
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/basicBlock/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/basicBlock/out.bin";
    shape inShape {2,64,100,100};
//    shape outShape {2,64,100,100};
    shape outShape {2,128,50,50};
    
    id <MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    
    id<MTLCommandBuffer> commandBuffer = block.forward(inputBuffer, inShape, outputBuffer, nullptr);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);

    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}


void testPreLayer(){
    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/preLayer/testData.json";
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *error;
    NSDictionary *layerInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
    
//    basicBlock block = makingBasicBlock(resource, basicBlockInfo);
    preLayer layer = makingPreLayer(resource, layerInfo);
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/preLayer/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/preLayer/out.bin";
    shape inShape {2,3,128,128};
//    shape outShape {2,64,100,100};
    shape outShape {2,64,32,32};
//    shape outShape {2,64,50,50};
    
    id <MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    
//    layer.

    id<MTLCommandBuffer> commandBuffer = layer.forward(inputBuffer, inShape, outputBuffer, nullptr);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);

    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}

void testResNet(){
    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/resnet/testData.json";
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *error;
    NSDictionary *netInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
    resnet net = makingResNet(resource, netInfo);
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/out.bin";
    shape inShape {17,3,256,256};
//    shape outShape {2,64,100,100};
//    shape outShape {2,128,16,16};
//    shape outShape {2,256,8,8};
//    shape outShape {4,512,8,8};
    shape outShape{17,1000,1,1};
    
    id <MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    
    id<MTLCommandBuffer> commandBuffer = net.forward(inputBuffer, inShape, outputBuffer, nullptr);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);

    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
    
}


void test_matmul(){
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/out.bin";
    
    std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/weight.bin";
    std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/bias.bin";
    
    shape inShape{1,512,1,1};
    shape outShape {1,1000,1,1};
    id<MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.channel);
    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    matmul mm(resource, inShape.channel, outShape.channel);
    tensor weight(1,1,outShape.channel, inShape.channel);
    tensor bias(1,outShape.channel,1,1);
    weight.loadFromFile(weight_path.c_str());
    bias.loadFromFile(bias_path.c_str());
    std::map<std::string, tensor> weights = {
        {"weight", std::move(weight)},
        {"bias", std::move(bias)}
    };
    mm.loadWeight(weights);
    matmulConstant matmulConst {1, (int)divUp(inShape.channel, 4), (int)outShape.channel};
    
    std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};;
    for(int i = 0; i < 10000000; i++){
        mm.runOnce(inOutBuffers, &matmulConst);
    }

    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
}

void timeTest(){
    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/resnet/testData.json";
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *error;
    NSDictionary *netInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
    resnet net = makingResNet(resource, netInfo);
    std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/input.bin";
    std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/out.bin";
    shape inShape {1,3,256,256};
//    shape outShape {2,64,100,100};
//    shape outShape {2,128,16,16};
//    shape outShape {2,256,8,8};
//    shape outShape {4,512,8,8};
    shape outShape{1,1000,1,1};
    
    id <MTLBuffer> inputBuffer = makingInputBuffer(input_path, inShape);
    id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    uint64_t startTime, stopTime;
    NSDate *methodStart = [NSDate date];
    for(int i = 0; i < 500; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = net.forward(inputBuffer, inShape, outputBuffer, nullptr);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
 
    }
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSLog(@"executionTime = %f", executionTime);
    
    
    
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);

    tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
    diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());

}

int main() {
//    test_conv();
//    test_bn();
//    test_act();
//    test_pooling();
//    test_elemWise();
//    test_convBnRelu();
//    testBufferPool();
//    testBasicBlock();
//    testPreLayer();
//    test_matmul();
//    testResNet();
    timeTest();

    return 0;
}
