#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

__device__ bool legendre_test_1(int n) {
    if (n <= 3) return n > 1;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_2(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_3(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_4(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_5(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_6(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_7(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_8(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_9(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ bool legendre_test_10(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void LegendreTestKernel(const int* numbers, bool* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch (rand() % 10) {
            case 0: results[idx] = legendre_test_1(numbers[idx]); break;
            case 1: results[idx] = legendre_test_2(numbers[idx]); break;
            case 2: results[idx] = legendre_test_3(numbers[idx]); break;
            case 3: results[idx] = legendre_test_4(numbers[idx]); break;
            case 4: results[idx] = legendre_test_5(numbers[idx]); break;
            case 5: results[idx] = legendre_test_6(numbers[idx]); break;
            case 6: results[idx] = legendre_test_7(numbers[idx]); break;
            case 7: results[idx] = legendre_test_8(numbers[idx]); break;
            case 8: results[idx] = legendre_test_9(numbers[idx]); break;
            case 9: results[idx] = legendre_test_10(numbers[idx]); break;
        }
    }
}

extern "C" void LegendreTest(const int* numbers, bool* results, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    LegendreTestKernel<<<blocksPerGrid, threadsPerBlock>>>(numbers, results, size);
}
