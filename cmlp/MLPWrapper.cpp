//
// Created by Prog on 27.03.2023.
//

#include "MLPWrapper.h"
#include "Layer.h"
#include <cmath>
#include <stdexcept>
#include <functional>
#include <list>
#include <vector>
#include <iostream>


MLPWrapper::MLPWrapper(float ***weights, const char** activationFunctionNames, int numLayers, int* numRows, int* numCols) {
    firstLayer = new Layer(weights[0], numRows[0], numCols[0], activationFunctionNames[0]);
    lastLayer = firstLayer;

    for (int i = 1; i < numLayers; i++) {
        auto nextLayer = new Layer(weights[i], numRows[i], numCols[i], activationFunctionNames[i]);
        lastLayer->setNextLayer(nextLayer);
        lastLayer = nextLayer;
    }
}

MLPWrapper::~MLPWrapper() {
    Layer* layer = firstLayer;
    while (layer) {
        Layer* nextLayer = layer->getNextLayer();
        delete layer;
        layer = nextLayer;
    }
}

float* MLPWrapper::forward(float* input) {
    return firstLayer->forward(input);
}

float* MLPWrapper::forwardUntil(int layerIndex, float* input) {
    if (layerIndex == 0) {
        return input;
    }
    else {
        Layer* untilLayer = firstLayer;
        for (size_t i = 0; i < layerIndex; i++) {
            untilLayer = untilLayer->getNextLayer();
        }
        return firstLayer->forwardUntil(untilLayer, input);
    }
}


//functional interface for the library
extern "C" MLPWrapper* __stdcall createMLPWrapper(float*** weights, const char **activationFunctionNames, int numLayers, int* numRows, int* numCols) {
    return new MLPWrapper(weights, activationFunctionNames, numLayers, numRows, numCols);
}

extern "C" void __stdcall deleteMLPWrapper(MLPWrapper *mlpWrapper) {
    delete mlpWrapper;
}

extern "C" float* __stdcall forward(MLPWrapper *mlpWrapper, float* input) {
    std::cout << "Input: " << input[0] << " " << input[1] << std::endl;
    float* data = mlpWrapper->forward(input);
    // create and return a pointer to the data
    return data;
}

extern "C" float* __stdcall forwardUntil(MLPWrapper *mlpWrapper, int layerIndex, float* input) {
    float* data = mlpWrapper->forwardUntil(layerIndex, input);
    // create and return a pointer to the data
    return data;
}



