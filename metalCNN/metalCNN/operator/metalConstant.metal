//
//  metalConstant.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/19.
//



struct convConstant{
    uint in_batch;
    uint in_slice; // slice per batch
    uint in_size; // size per slice, h*w
    uint in_height;
    uint in_width;
    
    uint out_batch;
    uint out_slice; // slice per batch
    uint out_size; //size per slice, h*w
    uint out_height;
    uint out_width;
    
    uint kernel_h;
    uint kernel_w;
    uint kernel_size;
    uint stride_h;
    uint stride_w;
    uint pad_x;
    uint pad_y;
};
