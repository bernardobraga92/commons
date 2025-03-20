#include <cuda_runtime.h>
#include <math.h>

__device__ int bernoulliNumber(int n) {
    if (n == 0) return 1;
    if (n % 2 == 0) return 0;
    int result = -((1 << (n - 1)) * ((n - 1) + 1));
    for (int k = 3; k < n; k += 2) {
        result /= k;
    }
    return result;
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int oddBernoulliPrime(int n) {
    int b = bernoulliNumber(n);
    if (b != 0 && isPrime(b)) return b;
    return -1;
}

__global__ void generateOddBernoulliPrimes(int *d_primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) d_primes[idx] = oddBernoulliPrime(idx);
}

extern "C" void run_oddbernoullicontconv(int *h_primes, int size) {
    int *d_primes;
    cudaMalloc((void**)&d_primes, size * sizeof(int));
    generateOddBernoulliPrimes<<<(size + 255) / 256, 256>>>(d_primes, size);
    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}
