//
// Created by Prog on 27.03.2023.
//

#include <cmath>
#include <cstring>
#include <iostream>
#include "Layer.h"

float ReLU(float x) {
    return x > 0 ? x : 0;
}

float LogSoftmax(float x) {
    return log(exp(x) + 1);
}

typedef float (*ActivationFunction)(float);

ActivationFunction Layer::getActivationFunction(const char *activationFunctionName) {
    if (strcmp(activationFunctionName, "ReLU") == 0) {
        return ReLU;
    }
    else if (strcmp(activationFunctionName, "LogSoftmax") == 0) {
        return LogSoftmax;
    }
    else {
        return ReLU;
    }
}

Layer::Layer(float **weights, int numRows, int numCols, const char *activationFunctionName) {
    this->weights = weights;
    this->numRows = numRows;
    this->numCols = numCols;
    this->activation = getActivationFunction(activationFunctionName);
    this->nextLayer = nullptr;
}

void Layer::setNextLayer(Layer *next) {
    this->nextLayer = next;
}

Layer *Layer::getNextLayer() {
    return this->nextLayer;
}

float* Layer::forward(const float* input) {
    float* output = new float[numRows];
    for (int i = 0; i < numRows; i++) {
        float weightedSum = 0;
        for (int j = 0; j < numCols; j++) {
            weightedSum += input[j] * weights[i][j];
        }
        output[i] = activation(weightedSum);
    }
    if (nextLayer) {
        return nextLayer->forward(output);
    }
    else {
        return output;
    }
}

float* Layer::forwardUntil(Layer *untilLayer, float* input) {
    if (this == untilLayer) {
        return forward(input);
    }
    else {
        float output[numRows];
        for (int i = 0; i < numRows; i++) {
            float weightedSum = 0;
            for (int j = 0; j < numCols; j++) {
                weightedSum += input[j] * weights[i][j];
            }
            output[i] = activation(weightedSum);
        }
        return nextLayer->forwardUntil(untilLayer, output);
    }
}
