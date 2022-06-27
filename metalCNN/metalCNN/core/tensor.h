//
//  tensor.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/18.
//

#ifndef tensor_h
#define tensor_h

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

using uint = unsigned int;
struct shape;

struct shape {
    uint batch;
    uint channel;
    uint height;
    uint width;
    uint size() const{
        return batch*channel*height*width;
    }
    uint sizeNC4HW4() const {
        return batch * roundUp(channel, 4) * height * width;
    }
};

struct stride {
    uint s_b;
    uint s_c;
    uint s_h;
    uint s_w;
};

class tensor {
public:
    enum class interpOrder {
        NCHW = 0,
        NC4HW4
    };
    tensor() = default;
    tensor& operator = (const tensor&t);
    tensor (const shape& s);
    tensor(const tensor& t);
    tensor (tensor&& t);
    tensor(uint b, uint c, uint h, uint w);
    ~tensor();
    
    size_t absSize() const {
        return absSize_;
    };
    
    size_t memSize() const {
        return memSize_;
    }
    void loadFromMemory(float* p, interpOrder order);
    
    void loadFromFile(const char* path);
    const shape& getShape() const {
        return shape_;
    }
    float* getRawPointer() const {
        return p_;
    }
    float at(uint i_b, uint i_c, uint i_h, uint i_w);
    void reInterpret(interpOrder order);
    
private:
    void toNCHW();
    void toNC4HW4();
    float* atNCHW(uint i_b, uint i_c, uint i_h, uint i_w, float* p);
    float* atNC4HW4(uint i_b, uint i_c, uint i_h, uint i_w, float* p);
    
    interpOrder order_ = interpOrder::NCHW;
    float* p_ = nullptr;
    shape shape_;
    size_t absSize_; // abstract size, i.e., b * c * h * w
    size_t memSize_; // memory size, equal to abstract size if order is NCHW.
    stride stride_;
};

#endif /* tensor_hpp */
