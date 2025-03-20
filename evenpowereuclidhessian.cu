#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GPU_BLOCK_SIZE 256

__global__ void evenPowerEuclidHessianKernel(unsigned long long *d_primes, unsigned long long limit) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 1 && i <= limit) {
        bool isPrime = true;
        for (unsigned long long j = 2; j * j <= i; ++j) {
            if (i % j == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) d_primes[i] = i;
    }
}

extern "C" void evenPowerEuclidHessian(unsigned long long *h_primes, unsigned long long limit) {
    unsigned long long *d_primes;
    cudaMalloc(&d_primes, sizeof(unsigned long long) * (limit + 1));
    cudaMemset(d_primes, 0, sizeof(unsigned long long) * (limit + 1));

    evenPowerEuclidHessianKernel<<<(limit + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(d_primes, limit);
    cudaMemcpy(h_primes, d_primes, sizeof(unsigned long long) * (limit + 1), cudaMemcpyDeviceToHost);

    cudaFree(d_primes);
}
