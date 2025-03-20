#include <stdio.h>
#include <math.h>

__device__ __host__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findLargePrimesKernel(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__global__ void fibonacciKernel(unsigned long long *fibonacci, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        fibonacci[0] = 0;
    } else if (idx == 1) {
        fibonacci[1] = 1;
    } else {
        fibonacci[idx] = fibonacci[idx - 1] + fibonacci[idx - 2];
    }
}

__global__ void compositeFibonacciKernel(unsigned long long *fibonacci, int *composites, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(fibonacci[idx])) {
        composites[idx] = fibonacci[idx];
    } else {
        composites[idx] = 0;
    }
}

__global__ void polarToCartesianKernel(float *r, float *theta, float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = r[idx] * cos(theta[idx]);
        y[idx] = r[idx] * sin(theta[idx]);
    }
}

__global__ void generatePrimesKernel(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__global__ void filterCompositesKernel(int *numbers, int *result, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(numbers[idx])) {
        result[idx] = numbers[idx];
    } else {
        result[idx] = 0;
    }
}

__global__ void cartesianToPolarKernel(float *x, float *y, float *r, float *theta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        r[idx] = sqrt(x[idx] * x[idx] + y[idx] * y[idx]);
        theta[idx] = atan2(y[idx], x[idx]);
    }
}

__global__ void primeFibonacciKernel(unsigned long long *fibonacci, int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(fibonacci[idx])) {
        primes[idx] = fibonacci[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void generateCompositesKernel(int *composites, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(idx)) {
        composites[idx] = idx;
    } else {
        composites[idx] = 0;
    }
}

__global__ void filterPrimesKernel(int *numbers, int *result, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(numbers[idx])) {
        result[idx] = numbers[idx];
    } else {
        result[idx] = 0;
    }
}

__global__ void fibonacciPrimesKernel(unsigned long long *fibonacci, int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(fibonacci[idx])) {
        primes[idx] = fibonacci[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void compositePrimesKernel(int *composites, int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(composites[idx])) {
        primes[idx] = composites[idx];
    } else {
        primes[idx] = 0;
    }
}

__global__ void cartesianToPolarKernel2(float *x, float *y, float *r, float *theta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        r[idx] = sqrt(x[idx] * x[idx] + y[idx] * y[idx]);
        theta[idx] = atan2(y[idx], x[idx]);
    }
}

__global__ void polarToCartesianKernel2(float *r, float *theta, float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = r[idx] * cos(theta[idx]);
        y[idx] = r[idx] * sin(theta[idx]);
    }
}

__global__ void generatePrimesKernel2(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

__global__ void filterCompositesKernel2(int *numbers, int *result, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(numbers[idx])) {
        result[idx] = numbers[idx];
    } else {
        result[idx] = 0;
    }
}

__global__ void cartesianToPolarKernel3(float *x, float *y, float *r, float *theta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        r[idx] = sqrt(x[idx] * x[idx] + y[idx] * y[idx]);
        theta[idx] = atan2(y[idx], x[idx]);
    }
}

__global__ void polarToCartesianKernel3(float *r, float *theta, float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = r[idx] * cos(theta[idx]);
        y[idx] = r[idx] * sin(theta[idx]);
    }
}

__global__ void generateCompositesKernel2(int *composites, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(idx)) {
        composites[idx] = idx;
    } else {
        composites[idx] = 0;
    }
}

__global__ void filterPrimesKernel2(int *numbers, int *result, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && isPrime(numbers[idx])) {
        result[idx] = numbers[idx];
    } else {
        result[idx] = 0;
    }
}

__global__ void fibonacciCompositesKernel(unsigned long long *fibonacci, int *composites, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && !isPrime(fibonacci[idx])) {
        composites[idx] = fibonacci[idx];
    } else {
        composites[idx] = 0;
    }
}
