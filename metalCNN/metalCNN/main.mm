//
//  main.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//


#import <Foundation/Foundation.h>
#import <AppKit/NSImage.h>
#import <AppKit/NSColor.h>
#include <chrono>
#include <iostream>
#include <string>
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "pooling.h"
#include "elemWise.h"
#include "matmul.h"
#include "resnet.h"
#include "builder.h"
#include "interp.h"
#include "testUtils.h"
#include "classes.h"

int main(int argc, char* argv[]) {
    NSImage* image = [[NSImage alloc] initWithContentsOfFile: @"/Users/tinglyfeng/Desktop/metalCNN/metalCNN/metalCNN/resource/lion.jpg"];
    
    std::map<int, std::string> classes {
        CLASSDICT
    };
    
    NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/resnet/ResNet.json";
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *error;
    NSDictionary *netInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
    
    NSBitmapImageRep* imageRep = [[NSBitmapImageRep alloc] initWithData:[image TIFFRepresentation]];
    unsigned char*  imageRawData =   [imageRep bitmapData];
    
    NSInteger samplesPerPixel = [imageRep samplesPerPixel];
    NSInteger bytesPerPlane = [imageRep bytesPerPlane];
    NSInteger bytesPerRow = [imageRep bytesPerRow];
    NSInteger numberOfPlanes = [imageRep numberOfPlanes];
    NSUInteger numPixels = numberOfPlanes * bytesPerPlane / samplesPerPixel;
    NSUInteger width = bytesPerRow / samplesPerPixel;
    NSUInteger height = numPixels / width;
    
    std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
    resnet net = makingResNet(resource, netInfo);
    
    id <MTLBuffer> inputBuffer = resource->getBuffer(numPixels * 4 * sizeof(float));
    float* floatImageData = (float*)inputBuffer.contents;
    for(int i = 0; i < numPixels; ++i){
        floatImageData[i*4] = (float( imageRawData[i*3+2]) / 255 - 0.406) / 0.225;
        floatImageData[i*4+1] = (float(imageRawData[i*3+1] )/ 255 - 0.456) / 0.224;
        floatImageData[i*4+2] = (float(imageRawData[i*3])/ 255 - 0.485) / 0.229;
    }
    
    shape inShape {1,3,(uint)height, (uint)width};
    shape outShape{1,1000,1,1};
    
    id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    id<MTLCommandBuffer> commandBuffer = net.forward(inputBuffer, inShape, outputBuffer, nullptr);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);
    
    float* p = metalOutTensor.getRawPointer();
    int maxIdx = -1;
    float maxAct = -10000000;
    for(int i = 0; i < 1000; i++){
        if(p[i] > maxAct){
            maxAct = p[i];
            maxIdx = i;
        }
    }
    std::string className = classes[maxIdx];
    std::cout <<  "Category of this image is : " << className << std::endl;
    
    
    return 0;
}
