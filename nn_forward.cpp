#include "nn_forward.h"
#include "weights.h"   // W1[64][784], b1[64], W2[10][64], b2[10] — BRAM ROMs

void nn_forward(
    float x[INPUT_SIZE],
    float y[OUTPUT_SIZE]
) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=DATA depth=784
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=DATA depth=10
#pragma HLS INTERFACE s_axilite port=return

    // Partition weight ROMs into 8 banks → 8 parallel reads per cycle
#pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=b1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=b2 complete dim=0

    float x_local[INPUT_SIZE];
    float h[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=h       complete dim=0

    // Load input from DDR — single 3 KB burst, done once
    LOAD_X: for (int j = 0; j < INPUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
        x_local[j] = x[j];
    }

    // Layer 1: h = ReLU(W1 * x + b1)
    // 8 independent partial accumulators break the FP dependency chain.
    // Loop steps j+=8 → 98 iterations. W1 and x_local each have 8 BRAM banks
    // (cyclic factor=8), so all 8 reads in each iteration hit different banks.
    // HLS achieves II=1 because consecutive updates to pacc[k] are 8 cycles apart,
    // which is ≥ the FP add latency.
    L1_row: for (int i = 0; i < HIDDEN_SIZE; i++) {
        float pacc[8] = {0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=pacc complete dim=0
        L1_col: for (int j = 0; j < INPUT_SIZE; j += 8) {
#pragma HLS PIPELINE II=1
            pacc[0] += W1[i][j+0] * x_local[j+0];
            pacc[1] += W1[i][j+1] * x_local[j+1];
            pacc[2] += W1[i][j+2] * x_local[j+2];
            pacc[3] += W1[i][j+3] * x_local[j+3];
            pacc[4] += W1[i][j+4] * x_local[j+4];
            pacc[5] += W1[i][j+5] * x_local[j+5];
            pacc[6] += W1[i][j+6] * x_local[j+6];
            pacc[7] += W1[i][j+7] * x_local[j+7];
        }
        float acc = b1[i] + pacc[0] + pacc[1] + pacc[2] + pacc[3]
                          + pacc[4] + pacc[5] + pacc[6] + pacc[7];
        h[i] = (acc > 0.0f) ? acc : 0.0f;
    }

    // Layer 2: y = W2 * h + b2
    // h is fully in registers. Same 8-accumulator pattern over HIDDEN_SIZE=64 (8 iterations).
    L2_row: for (int i = 0; i < OUTPUT_SIZE; i++) {
        float pacc[8] = {0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=pacc complete dim=0
        L2_col: for (int j = 0; j < HIDDEN_SIZE; j += 8) {
#pragma HLS PIPELINE II=1
            pacc[0] += W2[i][j+0] * h[j+0];
            pacc[1] += W2[i][j+1] * h[j+1];
            pacc[2] += W2[i][j+2] * h[j+2];
            pacc[3] += W2[i][j+3] * h[j+3];
            pacc[4] += W2[i][j+4] * h[j+4];
            pacc[5] += W2[i][j+5] * h[j+5];
            pacc[6] += W2[i][j+6] * h[j+6];
            pacc[7] += W2[i][j+7] * h[j+7];
        }
        y[i] = b2[i] + pacc[0] + pacc[1] + pacc[2] + pacc[3]
                     + pacc[4] + pacc[5] + pacc[6] + pacc[7];
    }
}
