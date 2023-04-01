//
// Created by Prog on 27.03.2023.
//

#ifndef CMLP_MLPWRAPPER_H
#define CMLP_MLPWRAPPER_H

#include <cmath>
#include <stdexcept>
#include <functional>
#include <list>
#include <vector>
#include <tuple>
#include "Layer.h"

class MLPWrapper {
private:
    Layer* firstLayer;
    Layer* lastLayer;

public:
    MLPWrapper(float*** weights, const char **activationFunctionNames, int numLayers, int* numRows, int* numCols);

    ~MLPWrapper();

    float* forward(float* input);

    float* forwardUntil(int layerIndex, float* input);
};


extern "C"  MLPWrapper* createMLPWrapper(float*** weights, const char **activationFunctionNames, int numLayers, int* numRows, int* numCols);
extern "C"  void deleteMLPWrapper(MLPWrapper* mlpWrapper);
extern "C"  float* forward(MLPWrapper* mlpWrapper, float* input);
extern "C"  float* forwardUntil(MLPWrapper* mlpWrapper, size_t layerIndex, float* input);


#endif //CMLP_MLPWRAPPER_H
