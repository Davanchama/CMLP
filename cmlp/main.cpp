//
// Created by Prog on 31.03.2023.
//
#include "MLPWrapper.h"
#include <iostream>

int main();

int main() {
    int numLayers = 3;
    float** weights1 = new float*[2];
    float** weights2 = new float*[2];
    float** weights3 = new float*[2];
    weights1[0] = new float[2]{1, 2};
    weights1[1] = new float[2]{3, 4};
    weights2[0] = new float[2]{5, 6};
    weights2[1] = new float[2]{7, 8};
    weights3[0] = new float[2]{9, 10};
    weights3[1] = new float[2]{11, 12};
    float*** weights = new float**[3]{weights1, weights2, weights3};
    const char* activationFunctionNames[3] = {"ReLU", "ReLU", "ReLU"};
    MLPWrapper* mlpWrapper = createMLPWrapper(weights, activationFunctionNames, numLayers, new int[3]{2, 2, 2}, new int[3]{2, 2, 2});
    float* output = forward(mlpWrapper, new float[2]{100, 100});
    std::cout << "Output: " << output[0] << " " << output[1] << std::endl;
    return 0;
}
