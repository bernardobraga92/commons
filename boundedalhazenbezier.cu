#include <cuda_runtime.h>
#include <math.h>

__global__ void boundedAlHazenBezierKernel(unsigned long long* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long n = numbers[idx];
    bool isPrime = true;

    for (unsigned long long i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            isPrime = false;
            break;
        }
    }

    numbers[idx] = isPrime ? n : 0;
}

extern "C" void boundedAlHazenBezier(unsigned long long* d_numbers, int size) {
    boundedAlHazenBezierKernel<<<(size + 255) / 256, 256>>>(d_numbers, size);
    cudaDeviceSynchronize();
}
