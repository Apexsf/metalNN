//
//  utils.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/18.
//

#include "utils.h"

void readDataFromFile(const char* path, size_t size, void* dst) {
    std::ifstream input;
    input.open(path, std::ios::in | std::ios::binary);
    input.read((char*)dst, size);
    input.close();
}
 
