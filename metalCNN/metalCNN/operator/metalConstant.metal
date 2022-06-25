//
//  metalConstant.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/19.
//

#ifndef metalConstant_metal
#define metalConstant_metal

struct convConstant{
    int in_batch;
    int in_slice; // slice per batch
    int in_size;  // size per slice, h*w
    int in_height;
    int in_width;
    
    int out_batch;
    int out_slice; // slice per batch
    int out_size;  //size per slice, h*w
    int out_height;
    int out_width;
    
    int kernel_h;
    int kernel_w;
    int kernel_size;
    int stride_h;
    int stride_w;
    int pad_x;
    int pad_y;
};


struct bnConstant {
    int batch;
    int slice;
    int height;
    int width;
    int size;
};

struct actConstant {
    int batch;
    int slice;
    int height;
    int width;
};

struct poolingConstant{
    int in_batch;
    int in_slice; // slice per batch
    int in_size;  // size per slice, h*w
    int in_height;
    int in_width;
    
    int out_batch;
    int out_slice; // slice per batch
    int out_size;  //size per slice, h*w
    int out_height;
    int out_width;
    
    int kernel_h;
    int kernel_w;
    int kernel_size;
    int stride_h;
    int stride_w;
    int pad_x;
    int pad_y;
};

#endif
