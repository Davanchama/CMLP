//
// Created by Prog on 27.03.2023.
//

#ifndef CMLP_LAYER_H
#define CMLP_LAYER_H

#include <vector>
#include <functional>
#include <string>
#include <list>

class Layer {
private:
    float** weights;
    std::function<float(float)> activation;
    Layer* nextLayer;
    typedef float (*ActivationFunction)(float);
    static ActivationFunction getActivationFunction(const char *activationFunctionName);

public:
    Layer(float **weights, int numRows, int numCols, const char *activationFunctionName);
    void setNextLayer(Layer* nextLayer);
    Layer* getNextLayer();
    float* forward(const float* input);
    float* forwardUntil(Layer* untilLayer, float* input);

    int numRows;
    int numCols;
};

#endif //CMLP_LAYER_H
