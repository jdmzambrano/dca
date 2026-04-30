#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[784],
    result_t layer4_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer4_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<_0_weight_t, 50176>(w2, "w2.txt");
        nnet::load_weights_from_txt<_0_bias_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<_2_weight_t, 640>(w4, "w4.txt");
        nnet::load_weights_from_txt<_2_bias_t, 10>(b4, "b4.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    _0_result_t layer2_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    nnet::dense<input_t, _0_result_t, config2>(input_1, layer2_out, w2, b2); // _0

    nnet::relu<_0_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // _1

    nnet::dense<layer3_t, result_t, config4>(layer3_out, layer4_out, w4, b4); // _2

}

