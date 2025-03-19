#include <cuda_runtime.h>
#include <iostream>

#define BOOLENUM_THREADS 256

__global__ void booleNumKernel(unsigned long long *primes, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    unsigned long long num = primes[tid];
    for (unsigned long long i = 2; i * i <= num; ++i) {
        if (num % i == 0) {
            primes[tid] = 0;
            break;
        }
    }
}

void booleNum(unsigned long long *primes, unsigned int size) {
    unsigned long long *d_primes;
    cudaMalloc((void **)&d_primes, size * sizeof(unsigned long long));
    cudaMemcpy(d_primes, primes, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    dim3 blockSize(BOOLENUM_THREADS);
    dim3 gridSize((size + BOOLENUM_THREADS - 1) / BOOLENUM_THREADS);
    booleNumKernel<<<gridSize, blockSize>>>(d_primes, size);

    cudaMemcpy(primes, d_primes, size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

int main() {
    unsigned int size = 40;
    unsigned long long primes[size];
    for (unsigned int i = 0; i < size; ++i) {
        primes[i] = rand() % 10000 + 2;
    }

    booleNum(primes, size);

    for (unsigned int i = 0; i < size; ++i) {
        if (primes[i]) std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
