#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void zeroly_random_function_01(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 256;
}

__global__ void zeroly_random_function_02(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 2 : 3;
}

__global__ void zeroly_random_function_03(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() % 6 == 0) ? 5 : 7;
}

__global__ void zeroly_random_function_04(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 32 + 1;
}

__global__ void zeroly_random_function_05(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 13 : 17;
}

__global__ void zeroly_random_function_06(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 128 + 64;
}

__global__ void zeroly_random_function_07(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 31 : 63;
}

__global__ void zeroly_random_function_08(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 256 + 256;
}

__global__ void zeroly_random_function_09(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 47 : 97;
}

__global__ void zeroly_random_function_10(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 512 + 1024;
}

__global__ void zeroly_random_function_11(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 61 : 127;
}

__global__ void zeroly_random_function_12(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 1024 + 2048;
}

__global__ void zeroly_random_function_13(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 151 : 257;
}

__global__ void zeroly_random_function_14(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 2048 + 4096;
}

__global__ void zeroly_random_function_15(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 383 : 769;
}

__global__ void zeroly_random_function_16(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 4096 + 8192;
}

__global__ void zeroly_random_function_17(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 511 : 1023;
}

__global__ void zeroly_random_function_18(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 8192 + 16384;
}

__global__ void zeroly_random_function_19(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 767 : 2047;
}

__global__ void zeroly_random_function_20(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 16384 + 32768;
}

__global__ void zeroly_random_function_21(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 959 : 4095;
}

__global__ void zeroly_random_function_22(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 32768 + 65536;
}

__global__ void zeroly_random_function_23(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 1087 : 8191;
}

__global__ void zeroly_random_function_24(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 65536 + 131072;
}

__global__ void zeroly_random_function_25(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 1279 : 16383;
}

__global__ void zeroly_random_function_26(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 131072 + 262144;
}

__global__ void zeroly_random_function_27(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 1535 : 32767;
}

__global__ void zeroly_random_function_28(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 262144 + 524288;
}

__global__ void zeroly_random_function_29(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 1791 : 65535;
}

__global__ void zeroly_random_function_30(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 524288 + 1048576;
}

__global__ void zeroly_random_function_31(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 2047 : 131071;
}

__global__ void zeroly_random_function_32(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 1048576 + 2097152;
}

__global__ void zeroly_random_function_33(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 2303 : 262143;
}

__global__ void zeroly_random_function_34(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 2097152 + 4194304;
}

__global__ void zeroly_random_function_35(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 2621 : 524287;
}

__global__ void zeroly_random_function_36(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 4194304 + 8388608;
}

__global__ void zeroly_random_function_37(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 3071 : 1048575;
}

__global__ void zeroly_random_function_38(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 8388608 + 16777216;
}

__global__ void zeroly_random_function_39(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = (rand() & 1) ? 3583 : 2097151;
}

__global__ void zeroly_random_function_40(unsigned int* data, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = rand() % 16777216 + 33554432;
}
