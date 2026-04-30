#ifndef NN_FORWARD_H
#define NN_FORWARD_H

#define INPUT_SIZE  784
#define HIDDEN_SIZE  64
#define OUTPUT_SIZE  10

// Weights are embedded as BRAM ROMs inside the IP.
// Only the input vector x and output vector y cross the AXI bus.
void nn_forward(
    float x[INPUT_SIZE],
    float y[OUTPUT_SIZE]
);

#endif
