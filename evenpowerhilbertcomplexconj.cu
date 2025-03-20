#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ inline bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
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

__global__ void hilbertCurve(int* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        numbers[idx] = x * x + y * y;
    }
}

__global__ void complexConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        imag[idx] = -imag[idx];
    }
}

__global__ void evenPower(int* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        numbers[idx] = pow(numbers[idx], 2);
    } else {
        numbers[idx] = 0;
    }
}

__global__ void hilbertComplexConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] += h;
        imag[idx] -= h;
    }
}

__global__ void evenPowerHilbert(float* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        numbers[idx] = pow(numbers[idx], 2) + h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void complexHilbertConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] += h;
        imag[idx] -= h;
    }
}

__global__ void evenPowerHilbertComplex(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(numbers[idx], 2) + h;
        imag[idx] -= h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void hilbertEvenPower(float* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        numbers[idx] = pow(h, 2);
    } else {
        numbers[idx] = 0;
    }
}

__global__ void complexEvenPower(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        real[idx] = pow(real[idx], 2);
        imag[idx] = -pow(imag[idx], 2);
    }
}

__global__ void hilbertComplexEvenPower(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(h, 2) + pow(real[idx], 2);
        imag[idx] = -pow(imag[idx], 2);
    }
}

__global__ void evenPowerComplexHilbert(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(numbers[idx], 2) + h;
        imag[idx] -= h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void hilbertEvenPowerComplex(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        numbers[idx] = pow(h, 2);
        imag[idx] -= h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void evenPowerHilbertComplexConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(numbers[idx], 2) + h;
        imag[idx] -= h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void hilbertComplexEvenPowerConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(h, 2);
        imag[idx] -= -pow(imag[idx], 2);
    } else {
        numbers[idx] = 0;
    }
}

__global__ void evenPowerHilbertComplexConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 == 0) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(numbers[idx], 2) + h;
        imag[idx] -= h;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void hilbertComplexEvenPowerConjugate(float* real, float* imag, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx % 2;
        int y = idx / 2;
        int h = x * x + y * y;
        real[idx] = pow(h, 2);
        imag[idx] -= -pow(imag[idx], 2);
    } else {
        numbers[idx] = 0;
    }
}
