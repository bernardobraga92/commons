#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void oddPowerVonneumannInvHypTrigKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned long long num = d_numbers[idx];
    d_numbers[idx] = ((num * num * num) % 97) + 1; // Example operation
}

__global__ void isPrimeKernel(unsigned long long* d_numbers, int* d_isPrime, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned long long num = d_numbers[idx];
    bool prime = true;
    for (unsigned long long i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) {
            prime = false;
            break;
        }
    }
    d_isPrime[idx] = prime ? 1 : 0;
}

__global__ void generatePrimesKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] = idx * idx + idx + 41; // Example polynomial generating primes
}

__global__ void filterPrimesKernel(unsigned long long* d_numbers, int* d_isPrime, unsigned long long* d_filtered, int* d_count, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || d_isPrime[idx] == 0) return;

    int globalIdx = atomicAdd(d_count, 1);
    d_filtered[globalIdx] = d_numbers[idx];
}

__global__ void addOneKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] += 1;
}

__global__ void multiplyByTwoKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] *= 2;
}

__global__ void subtractOneKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] -= 1;
}

__global__ void divideByTwoKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] /= 2;
}

__global__ void moduloOperationKernel(unsigned long long* d_numbers, unsigned long long divisor, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] %= divisor;
}

__global__ void bitwiseAndKernel(unsigned long long* d_numbers, unsigned long long mask, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] &= mask;
}

__global__ void bitwiseOrKernel(unsigned long long* d_numbers, unsigned long long mask, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] |= mask;
}

__global__ void bitwiseXorKernel(unsigned long long* d_numbers, unsigned long long mask, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] ^= mask;
}

__global__ void shiftLeftKernel(unsigned long long* d_numbers, unsigned long long shift, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] <<= shift;
}

__global__ void shiftRightKernel(unsigned long long* d_numbers, unsigned long long shift, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] >>= shift;
}

__global__ void addTwoKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] += 2;
}

__global__ void multiplyByThreeKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] *= 3;
}

__global__ void subtractTwoKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] -= 2;
}

__global__ void divideByThreeKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] /= 3;
}

__global__ void bitwiseNotKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] = ~d_numbers[idx];
}

__global__ void sqrtOperationKernel(unsigned long long* d_numbers, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    d_numbers[idx] = static_cast<unsigned long long>(sqrt(d_numbers[idx]));
}
