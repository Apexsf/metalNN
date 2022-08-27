//
//  convBn.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/8/27.
//

#include "convBn.h"

convBn::convBn(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params) : conv(resource, name, params){
    
}

void convBn::loadWeight(std::map<std::string, tensor>& weights)  {
    uint ic = params_.inC;
    uint oc = params_.outC;
    uint h = params_.kernelH;
    uint w  = params_.kernelW;
    
    uint icDU4 = divUp(ic, 4);
    uint ocDU4 = divUp(oc, 4);
    
//    uint ocRU4 = roundUp(ocDU4, 4);
//    size_t dstSize = ocRU4 * icDU4 * 16 * h * w;
    size_t dstSize = ocDU4 * icDU4 * 16 * h * w;
    
    weight_ = makingBuffer(dstSize * sizeof(float), MTLResourceStorageModeShared);
    
    float* src_p = weights["weight"].getRawPointer();
    float* dst_p = (float*) weight_.contents;
    
    float* bnRM = weights["running_mean"].getRawPointer(); // bn weights
    float* bnRV = weights["running_var"].getRawPointer(); // bn weights
    float* bnGamma = weights["gamma"].getRawPointer(); // bn weights
    float* bnBeta = weights["beta"].getRawPointer(); // bn weights
    
    for(uint i_oc = 0; i_oc< oc; ++i_oc){
        uint dst_o_order = i_oc  / 4;
        uint dst_o_remainder = i_oc  % 4;
        float curBnScale = bnGamma[i_oc] / sqrt(bnRV[i_oc] + 0.00001);
        float* dst_o_p = dst_p + dst_o_order * icDU4 * 16 * h * w + dst_o_remainder * 4;
        for(uint i_ic = 0; i_ic < ic; ++i_ic) {
            uint dst_i_order = i_ic  / 4;
            uint dst_i_remainder = i_ic  % 4;
            float* dst_i_p = dst_o_p + dst_i_order * 16 * h * w + dst_i_remainder;
            for(uint i_h = 0; i_h < h; i_h++){
                for(uint i_w = 0; i_w < w; i_w++){
                    dst_i_p[(i_h * w + i_w) * 16] = (*src_p) * curBnScale; // scale with bn weights
                    src_p++;
                }
            }
        }
    }
    
    // load bias
    uint outCDU = roundUp(params_.outC, 4);
    bias_ = makingBuffer(outCDU * sizeof(float), MTLResourceStorageModeShared);
    float* dst_b_p = (float*) bias_.contents;
    if (weights.find("bias") != weights.end()) {
        float* src_b_p = weights["bias"].getRawPointer();
        for(uint i = 0; i < oc; i++){
            float curBnScale = bnGamma[i] / sqrt(bnRV[i] + 0.00001);
            float curBnBias =  - bnRM[i] / sqrt(bnRV[i] + 0.00001) * bnGamma[i] + bnBeta[i];
            // note: bias should be scaled and biased both by bn params
            dst_b_p[i] = src_b_p[i] * curBnScale + curBnBias;
        }
    }else {
        for(uint i = 0; i < oc; i++){
            float curBnBias =  - bnRM[i] / sqrt(bnRV[i] + 0.00001) * bnGamma[i] + bnBeta[i];
            dst_b_p[i] = curBnBias;
        }
    }
    for(uint i = oc; i < outCDU; i++){
        dst_b_p[i] = 0;
    }
}
