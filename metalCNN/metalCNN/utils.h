//
//  utils.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/18.
//

#ifndef utils_h
#define utils_h

#include <stdio.h>
#include <iostream>
#include <fstream>

void readDataFromFile(const char* path, size_t size, void* dst);

inline uint divUp(uint src, uint div){
    return (src + div - 1) / div;
}

inline uint roundUp(uint src, uint div) {
    return divUp(src, div) * div;
}

#endif /* utils_hpp */
