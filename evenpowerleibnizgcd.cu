#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesEvenPowerLeibnizGCD(int *d_primes, int limit) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < limit) {
        if (isPrime(idx) && isPrime(pow(2, idx) + 1)) {
            d_primes[idx] = idx;
        }
        idx += gridDim.x * blockDim.x;
    }
}

int main() {
    int limit = 1000;
    int *h_primes = new int[limit];
    int *d_primes;

    cudaMalloc(&d_primes, sizeof(int) * limit);
    findPrimesEvenPowerLeibnizGCD<<<256, 256>>>(d_primes, limit);

    cudaMemcpy(h_primes, d_primes, sizeof(int) * limit, cudaMemcpyDeviceToHost);

    for (int i = 0; i < limit; ++i) {
        if (h_primes[i] != 0) {
            std::cout << h_primes[i] << " ";
        }
    }

    delete[] h_primes;
    cudaFree(d_primes);
    return 0;
}
