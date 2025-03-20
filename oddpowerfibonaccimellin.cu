#include <iostream>
#include <cmath>

__global__ void gpuOddPowerFibonacciMellin(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = pow(data[idx], 3); // Example transformation
    }
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

__global__ void gpuFindLargePrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] *= data[idx]; // Example transformation on primes
    }
}

__global__ void gpuFibonacci(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (idx == 0) data[idx] = 0;
        else if (idx == 1) data[idx] = 1;
        else data[idx] = data[idx - 1] + data[idx - 2];
    }
}

__global__ void gpuMellinTransform(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= log(data[idx]); // Example transformation
    }
}

// Additional functions

__global__ void gpuOddPower(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] % 2 != 0) {
        data[idx] = pow(data[idx], 3);
    }
}

__global__ void gpuEvenSquareRoot(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] % 2 == 0) {
        data[idx] = sqrt(data[idx]);
    }
}

__global__ void gpuIncrementPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx]++;
    }
}

__global__ void gpuDecrementNonPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !isPrime(data[idx])) {
        data[idx]--;
    }
}

__global__ void gpuSquareFibonacci(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= data[idx];
    }
}

__global__ void gpuCubeNonPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !isPrime(data[idx])) {
        data[idx] = pow(data[idx], 3);
    }
}

__global__ void gpuAddTenToOdds(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] % 2 != 0) {
        data[idx] += 10;
    }
}

__global__ void gpuSubtractFiveFromEvens(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] % 2 == 0) {
        data[idx] -= 5;
    }
}

__global__ void gpuMultiplyBySevenPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] *= 7;
    }
}

__global__ void gpuDivideByTwoNonPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !isPrime(data[idx]) && data[idx] > 0) {
        data[idx] /= 2;
    }
}

__global__ void gpuNegateLargeNumbers(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1000) {
        data[idx] *= -1;
    }
}

__global__ void gpuCapPrimesAtFiveHundred(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx]) && data[idx] > 500) {
        data[idx] = 500;
    }
}

__global__ void gpuDoubleNonPrimes(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !isPrime(data[idx])) {
        data[idx] *= 2;
    }
}
