#include <iostream>
#include <cmath>

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__device__ unsigned long long int transcendentalBesselContConv(unsigned long long int x, int depth) {
    if (depth == 0) return x;
    return transcendentalBesselContConv(x * 2 + depth, depth - 1);
}

__global__ void findLargePrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long int candidate = transcendentalBesselContConv(idx * 137 + 98547, 10);
        if (isPrime(candidate)) primes[idx] = candidate;
        else primes[idx] = -1;
    }
}

int main() {
    const int size = 256;
    int* d_primes;
    cudaMalloc((void**)&d_primes, size * sizeof(int));

    findLargePrimes<<<(size + 255) / 256, 256>>>(d_primes, size);

    int* h_primes = new int[size];
    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (h_primes[i] != -1) std::cout << "Prime: " << h_primes[i] << std::endl;
    }

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
