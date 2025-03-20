#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isRationalHermite(int num) {
    return isPrime(num);
}

__global__ void filterRationalHermite(int* numbers, int* filtered, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isRationalHermite(numbers[idx])) {
        filtered[idx] = numbers[idx];
    } else {
        filtered[idx] = 0;
    }
}

__device__ bool isGradientPrime(int num) {
    return isPrime(num);
}

__global__ void findGradientPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isGradientPrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isHermitePrime(int num) {
    return isPrime(num);
}

__global__ void findHermitePrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isHermitePrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isRationalPrime(int num) {
    return isPrime(num);
}

__global__ void findRationalPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isRationalPrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isGradHermite(int num) {
    return isPrime(num);
}

__global__ void findGradHermitePrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isGradHermite(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isPrimeGrad(int num) {
    return isPrime(num);
}

__global__ void findPrimeGradPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrimeGrad(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isHermiteGrad(int num) {
    return isPrime(num);
}

__global__ void findHermiteGradPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isHermiteGrad(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isRationalGrad(int num) {
    return isPrime(num);
}

__global__ void findRationalGradPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isRationalGrad(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isHermitePrimeGrad(int num) {
    return isPrime(num);
}

__global__ void findHermitePrimeGradPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isHermitePrimeGrad(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isRationalHermiteGrad(int num) {
    return isPrime(num);
}

__global__ void findRationalHermiteGradPrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isRationalHermiteGrad(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isPrimeRationalHermite(int num) {
    return isPrime(num);
}

__global__ void findPrimeRationalHermitePrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrimeRationalHermite(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

__device__ bool isGradHermitePrime(int num) {
    return isPrime(num);
}

__global__ void findGradHermitePrimePrimes(int* numbers, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isGradHermitePrime(numbers[idx])) {
        primes[idx] = numbers[idx];
    } else {
        primes[idx] = 0;
    }
}

int main() {
    int* h_numbers, *h_primes;
    int* d_numbers, *d_primes;

    h_numbers = (int*)malloc(N * sizeof(int));
    h_primes = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        h_numbers[i] = rand() % 10000;
    }

    cudaMalloc(&d_numbers, N * sizeof(int));
    cudaMalloc(&d_primes, N * sizeof(int));

    cudaMemcpy(d_numbers, h_numbers, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    findPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    filterRationalHermite<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findGradientPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findHermitePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findRationalPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findGradHermitePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findPrimeGradPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findHermiteGradPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findRationalGradPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findHermitePrimeGradPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findRationalHermiteGradPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findPrimeRationalHermitePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);
    findGradHermitePrimePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_numbers, d_primes, N);

    cudaMemcpy(h_primes, d_primes, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    free(h_numbers);
    free(h_primes);
    cudaFree(d_numbers);
    cudaFree(d_primes);

    return 0;
}
