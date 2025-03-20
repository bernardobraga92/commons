#include <cuda_runtime.h>
#include <stdio.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesVonneumannExp(int *d_primes, int limit, bool *foundPrime, int maxCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && *foundPrime == false) {
        if (isPrime(idx)) {
            d_primes[atomicAdd(&maxCount, 1)] = idx;
            *foundPrime = true;
        }
    }
}

__global__ void generatePrimesVonneumannExp(int *d_primes, int *d_maxCount, int limit) {
    __shared__ bool foundPrime;
    if (threadIdx.x == 0) foundPrime = false;
    __syncthreads();
    
    findPrimesVonneumannExp<<<(limit + 255) / 256, 256>>>(d_primes, limit, &foundPrime, d_maxCount[0]);
}

__host__ void executePrimesVonneumannExp(int limit) {
    int *h_primes = new int[limit];
    bool *h_foundPrime = new bool{false};
    int *h_maxCount = new int{0};

    int *d_primes, *d_foundPrime, *d_maxCount;
    cudaMalloc(&d_primes, sizeof(int) * limit);
    cudaMalloc(&d_foundPrime, sizeof(bool));
    cudaMalloc(&d_maxCount, sizeof(int));

    cudaMemcpy(d_foundPrime, h_foundPrime, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxCount, h_maxCount, sizeof(int), cudaMemcpyHostToDevice);

    generatePrimesVonneumannExp<<<1, 256>>>(d_primes, d_maxCount, limit);

    cudaMemcpy(h_primes, d_primes, sizeof(int) * limit, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_foundPrime, d_foundPrime, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxCount, d_maxCount, sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < *h_maxCount; ++i) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    delete[] h_primes;
    delete h_foundPrime;
    delete h_maxCount;

    cudaFree(d_primes);
    cudaFree(d_foundPrime);
    cudaFree(d_maxCount);
}

int main() {
    executePrimesVonneumannExp(1000);
    return 0;
}
