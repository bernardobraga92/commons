#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int generateRandomPrime(int seed, int lowerBound, int upperBound) {
    while (true) {
        int num = seed + (rand() % (upperBound - lowerBound + 1));
        if (isPrime(num)) return num;
    }
}

__device__ void evenPowerFourierDiscConvKernel(int *data, int size, int seed) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        data[i] = generateRandomPrime(seed + i, 1000, 10000);
    }
}

__global__ void evenPowerFourierDiscConv(int *data, int size, int seed) {
    evenPowerFourierDiscConvKernel(data, size, seed);
}

extern "C" void evenPowerFourierDiscConvLauncher(int *data, int size, int seed) {
    evenPowerFourierDiscConv<<<(size + 255) / 256, 256>>>(data, size, seed);
}
