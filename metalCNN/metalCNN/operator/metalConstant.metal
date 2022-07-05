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

struct elemWiseConstant {
    int batch;
    int slice;
    int height;
    int width;
    int size;
};

struct matmulConstant{
    int batch;
    int inSlice; // divUp(inC, 4)
    int outC;
};

struct  interpBilinearConstant{
    int in_batch;
    int in_slice;
    int in_size;
    int in_height;
    int in_width;
    
    int out_batch;
    int out_slice;
    int out_size;
    int out_height;
    int out_width;
    
    
// for bilinear interpolation, see
// https://stackoverflow.com/questions/70024313/resize-using-bilinear-interpolation-in-python
//    float w_scale_center;
//    float h_scale_center;
//    float w_scale;
//    float h_scale;
    
    float w_scale;
    float h_scale;
    float w_offset; // = -w_scale_center * w_scale+ w_ori_center
    float h_offset ; // = -h_scale_center * h_scale + h_ori_center
};

#endif
