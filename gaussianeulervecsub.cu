#include <curand.h>
#include <cmath>

__global__ void GaussianEulerVecSub1(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - cos(x);
    }
}

__global__ void GaussianEulerVecSub2(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void GaussianEulerVecSub3(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - sin(b[idx]);
    }
}

__global__ void GaussianEulerVecSub4(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub5(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - tan(x);
    }
}

__global__ void GaussianEulerVecSub6(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub7(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - sqrt(b[idx]);
    }
}

__global__ void GaussianEulerVecSub8(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub9(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - log(x);
    }
}

__global__ void GaussianEulerVecSub10(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub11(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - pow(b[idx], 2);
    }
}

__global__ void GaussianEulerVecSub12(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub13(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - floor(x);
    }
}

__global__ void GaussianEulerVecSub14(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub15(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - ceil(b[idx]);
    }
}

__global__ void GaussianEulerVecSub16(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub17(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - modf(x);
    }
}

__global__ void GaussianEulerVecSub18(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub19(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - frexp(b[idx]);
    }
}

__global__ void GaussianEulerVecSub20(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub21(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - ldexp(x);
    }
}

__global__ void GaussianEulerVecSub22(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub23(float* a, float* b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        a[idx] += exp(b[idx]) - erf(b[idx]);
    }
}

__global__ void GaussianEulerVecSub24(int* primes, int* candidates, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && isPrime(candidates[idx])) {
        primes[idx] = candidates[idx];
    }
}

__global__ void GaussianEulerVecSub25(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = exp(x) - erfc(x);
    }
}
