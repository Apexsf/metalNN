//
//  test.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#ifndef test_h
#define test_h

#include <string>
#include <Metal/Metal.h>
#include "tensor.h"
#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "pooling.h"
#include "elemWise.h"
#include "matmul.h"
#include "resnet.h"
#include "builder.h"
#include "interp.h"


void testConv();
void testBn();
void testAct();
void testPooling();
void testElemWise();
void testMatMul();
void testInterp();
void testResNet();


id<MTLBuffer> makingInputBuffer(std::shared_ptr<gpuResource>resource, std::string input_path, const shape& shp);


tensor makingTorchOutTensorNCHW(std::string output_path, const shape& shp) ;

tensor makingMetalOutTensorNCHW(id<MTLBuffer> outBuffer, const shape& shp);

void diffProfile(float* p1, float* p2, size_t size);

#endif /* test_h */
