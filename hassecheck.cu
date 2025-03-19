#ifndef HASSECHECK_H
#define HASSECHECK_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ __forceinline__ unsigned long long int hasseCheckFunction1(unsigned long long int n) {
    return (n * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction2(unsigned long long int n) {
    return (n * n - 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction3(unsigned long long int n) {
    return (n * n + n) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction4(unsigned long long int n) {
    return (n * n - n) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction5(unsigned long long int n) {
    return (n * n + n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction6(unsigned long long int n) {
    return (n * n - n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction7(unsigned long long int n) {
    return (n * n + 2 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction8(unsigned long long int n) {
    return (n * n - 2 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction9(unsigned long long int n) {
    return (n * n + 3 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction10(unsigned long long int n) {
    return (n * n - 3 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction11(unsigned long long int n) {
    return (n * n + 4 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction12(unsigned long long int n) {
    return (n * n - 4 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction13(unsigned long long int n) {
    return (n * n + 5 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction14(unsigned long long int n) {
    return (n * n - 5 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction15(unsigned long long int n) {
    return (n * n + 6 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction16(unsigned long long int n) {
    return (n * n - 6 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction17(unsigned long long int n) {
    return (n * n + 7 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction18(unsigned long long int n) {
    return (n * n - 7 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction19(unsigned long long int n) {
    return (n * n + 8 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction20(unsigned long long int n) {
    return (n * n - 8 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction21(unsigned long long int n) {
    return (n * n + 9 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction22(unsigned long long int n) {
    return (n * n - 9 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction23(unsigned long long int n) {
    return (n * n + 10 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction24(unsigned long long int n) {
    return (n * n - 10 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction25(unsigned long long int n) {
    return (n * n + 11 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction26(unsigned long long int n) {
    return (n * n - 11 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction27(unsigned long long int n) {
    return (n * n + 12 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction28(unsigned long long int n) {
    return (n * n - 12 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction29(unsigned long long int n) {
    return (n * n + 13 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction30(unsigned long long int n) {
    return (n * n - 13 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction31(unsigned long long int n) {
    return (n * n + 14 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction32(unsigned long long int n) {
    return (n * n - 14 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction33(unsigned long long int n) {
    return (n * n + 15 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction34(unsigned long long int n) {
    return (n * n - 15 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction35(unsigned long long int n) {
    return (n * n + 16 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction36(unsigned long long int n) {
    return (n * n - 16 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction37(unsigned long long int n) {
    return (n * n + 17 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction38(unsigned long long int n) {
    return (n * n - 17 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction39(unsigned long long int n) {
    return (n * n + 18 * n + 1) % 4294967291;
}

__device__ __forceinline__ unsigned long long int hasseCheckFunction40(unsigned long long int n) {
    return (n * n - 18 * n + 1) % 4294967291;
}
