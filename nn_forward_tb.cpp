#include <stdio.h>
#include <math.h>
#include "nn_forward.h"

static void read_bin(const char* path, float* buf, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return; }
    fread(buf, sizeof(float), n, f);
    fclose(f);
}

int main() {
    static float x[INPUT_SIZE];
    static float y[OUTPUT_SIZE];
    static float y_ref[OUTPUT_SIZE];

    read_bin("x.bin",           x,     INPUT_SIZE);
    read_bin("y_expected.bin",  y_ref, OUTPUT_SIZE);

    nn_forward(x, y);

    printf("Output logits:\n");
    int pred = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  class %d: HLS=%.4f  ref=%.4f\n", i, y[i], y_ref[i]);
        if (y[i] > y[pred]) pred = i;
    }
    printf("Predicted class: %d\n", pred);

    float max_err = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float err = fabsf(y[i] - y_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("Max absolute error vs reference: %.6f\n", max_err);
    if (max_err < 1e-3f)
        printf("PASS\n");
    else
        printf("FAIL\n");

    return 0;
}
