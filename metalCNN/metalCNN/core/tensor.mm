//
//  tensor.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/18.
//

#include "tensor.h"

inline size_t calSize(const shape& s) {
    return  s.batch * s.channel * s.height * s.width;
}

tensor::tensor(const shape& s) : shape_(s), stride_{s.channel * s.height * s.width, s.height * s.width, s.width, 1} {
    absSize_ = calSize(shape_);
    memSize_ = absSize_;
    p_ = (float*) malloc(absSize_ * sizeof(float));
}

tensor::tensor(uint b, uint c, uint h, uint w): shape_{b,c,h,w},
stride_{c*h*w, h*w, w, 1}{
    absSize_ = calSize(shape_);
    memSize_ = absSize_;
    p_ = (float*) malloc(absSize_ * sizeof(float));
}

tensor::~tensor(){
    if (p_) {
        free(p_);
    }
}

void tensor::loadFromMemory(float* p, interpOrder order){
    if (order_ == order) {
        memcpy(p_, p, memSize_ * sizeof(float));
    } else {
        order_ = order;
        size_t oldMemSize = memSize_;
        if (order == interpOrder::NCHW){
            memSize_ = absSize_;
        } else {
            memSize_ = shape_.batch * roundUp(shape_.channel, 4) * stride_.s_c;
        }
        if (oldMemSize != memSize_) {
            if(p_) free(p_);
             p_ = (float*)malloc(memSize_ * sizeof(float));
        }
        memcpy(p_, p, memSize_ * sizeof(float));
    }
}

void tensor::loadFromFile(const char* path) {
    if(p_){
        readDataFromFile(path, absSize_ * sizeof(float), (void*) p_);
    } else{
        std::cerr << "reading data to empty tensor!" << std::endl;
    }

}

float tensor::at(uint i_b, uint i_c, uint i_h, uint i_w) {
    if (order_ == interpOrder::NCHW) {
        return *atNCHW(i_b, i_c, i_h, i_w, p_);
    } else{
        return *atNC4HW4(i_b, i_c, i_h, i_w, p_);
    }
}



inline float* tensor::atNCHW(uint i_b, uint i_c, uint i_h, uint i_w, float* p){
    return p + i_b * stride_.s_b + i_c * stride_.s_c + i_h * stride_.s_h  + i_w * stride_.s_w;
}

inline float* tensor::atNC4HW4(uint i_b, uint i_c, uint i_h, uint i_w, float* p){
    return p +
        i_b * roundUp(shape_.channel, 4)  * stride_.s_c + (i_c / 4) * 4 * stride_.s_c  + i_h * stride_.s_h * 4 + i_w * 4 * stride_.s_w + i_c % 4
    ;
}

void tensor::reInterpret(tensor::interpOrder order) {
    if (order == order_) return;
    else if (order == interpOrder::NCHW) {
        toNCHW();
    } else {
        toNC4HW4();
    }
}

void tensor::toNCHW(){
    memSize_ = absSize_;
    float* new_p = (float*) malloc(memSize_ * sizeof(float));
    for(uint i_b = 0; i_b < shape_.batch; ++i_b){
        for(uint i_c = 0; i_c < shape_.channel; ++i_c){
            for(uint i_h = 0; i_h < shape_.height; ++i_h){
                for(uint i_w = 0; i_w < shape_.width; ++i_w){
                    *atNCHW(i_b, i_c, i_h, i_w, new_p) = *atNC4HW4(i_b, i_c, i_h, i_w, p_);
                }
            }
        }
    }
    free(p_);
    p_ = new_p;
    order_ = interpOrder::NCHW;
    
}

void tensor::toNC4HW4(){
    memSize_ =shape_.batch * roundUp(shape_.channel, 4) * stride_.s_c;
    float* new_p = (float*) malloc(memSize_ * sizeof(float));
    
    for(uint i_b = 0; i_b < shape_.batch; ++i_b){
        for(uint i_c = 0; i_c < shape_.channel; ++i_c){
            for(uint i_h = 0; i_h < shape_.height; ++i_h){
                for(uint i_w = 0; i_w < shape_.width; ++i_w){
                    *(atNC4HW4(i_b, i_c, i_h, i_w, new_p)) = *(atNCHW(i_b, i_c, i_h, i_w, p_));
 
                }
            }
        }
    }
    free(p_);
    p_ = new_p;
    order_ = interpOrder::NC4HW4;
    
}

