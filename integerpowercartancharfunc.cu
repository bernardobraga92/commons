#ifndef INTEGERPOWERCARTANCHARFUNC_H
#define INTEGERPOWERCARTANCHARFUNC_H

#include <cuda_runtime.h>
#include <math.h>

__device__ __inline__ unsigned int integer_power_cartan_char_func_0(unsigned int a, unsigned int b) {
    return a * b + 1;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_1(unsigned int a, unsigned int b) {
    return (a ^ b) + 3;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_2(unsigned int a, unsigned int b) {
    return (a / b) * 7 + 5;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_3(unsigned int a, unsigned int b) {
    return (a % b) * 11 + 9;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_4(unsigned int a, unsigned int b) {
    return (a & b) * 13 + 17;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_5(unsigned int a, unsigned int b) {
    return (a | b) * 19 + 23;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_6(unsigned int a, unsigned int b) {
    return (a << b) * 29 + 31;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_7(unsigned int a, unsigned int b) {
    return (a >> b) * 37 + 41;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_8(unsigned int a, unsigned int b) {
    return sqrt(a + b) * 43 + 47;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_9(unsigned int a, unsigned int b) {
    return (a - b) * 53 + 59;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_10(unsigned int a, unsigned int b) {
    return (a + b) * 61 + 67;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_11(unsigned int a, unsigned int b) {
    return (a * b) ^ 71 + 73;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_12(unsigned int a, unsigned int b) {
    return (a / b) | 79 + 83;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_13(unsigned int a, unsigned int b) {
    return (a % b) << 89 + 97;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_14(unsigned int a, unsigned int b) {
    return (a & b) >> 101 + 103;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_15(unsigned int a, unsigned int b) {
    return (a | b) - 107 + 109;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_16(unsigned int a, unsigned int b) {
    return (a ^ b) * 113 + 127;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_17(unsigned int a, unsigned int b) {
    return (a << b) / 131 + 137;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_18(unsigned int a, unsigned int b) {
    return (a >> b) % 139 + 149;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_19(unsigned int a, unsigned int b) {
    return (a & b) * 151 + 157;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_20(unsigned int a, unsigned int b) {
    return (a | b) ^ 163 + 167;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_21(unsigned int a, unsigned int b) {
    return (a - b) << 179 + 181;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_22(unsigned int a, unsigned int b) {
    return (a + b) >> 191 + 193;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_23(unsigned int a, unsigned int b) {
    return (a * b) - 197 + 199;
}

__device__ __inline__ unsigned int integer_power_cartan_char_func_24(unsigned int a, unsigned int b) {
    return (a / b) | 211 + 223;
}
