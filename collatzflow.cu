#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findLargePrimes(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < limit) {
        if (isPrime(idx)) primes[idx] = idx;
        idx += gridDim.x * blockDim.x;
    }
}

int main() {
    const int limit = 1000000;
    int *h_primes = new int[limit];
    memset(h_primes, 0, sizeof(int) * limit);
    int *d_primes;
    cudaMalloc(&d_primes, sizeof(int) * limit);

    findLargePrimes<<<128, 128>>>(d_primes, limit);

    cudaMemcpy(h_primes, d_primes, sizeof(int) * limit, cudaMemcpyDeviceToHost);
    cudaFree(d_primes);

    for (int i = 0; i < limit; ++i)
        if (h_primes[i]) std::cout << h_primes[i] << " ";

    delete[] h_primes;
    return 0;
}
