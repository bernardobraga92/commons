#include <cmath>
#include <curand_kernel.h>

__global__ void generatePrimes(int* primes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);

    int candidate = 2 + (curand(&state) % (n - 2));
    while (!isPrime(candidate)) {
        candidate = 2 + (curand(&state) % (n - 2));
    }
    primes[idx] = candidate;
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void complexConjugate(double* real, double* imag, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    imag[idx] = -imag[idx];
}

__global__ void naturalStieltjes(int* primes, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += log(primes[i]);
    }
    result[idx] = sum / n;
}

__global__ void randomGaussian(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);
    data[idx] = curand_normal(&state);
}

__global__ void matrixMultiply(double* A, double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void vectorAddition(double* a, double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = a[idx] + b[idx];
}

__global__ void scalarMultiplication(double* a, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    a[idx] *= scalar;
}

__global__ void elementwiseProduct(double* a, double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = a[idx] * b[idx];
}

__global__ void inverseSqrt(double* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    a[idx] = 1.0 / sqrt(a[idx]);
}

__global__ void powerFunction(double* a, double exponent, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = pow(a[idx], exponent);
}

__global__ void exponentialFunction(double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = exp(a[idx]);
}

__global__ void logarithmicFunction(double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = log(a[idx]);
}

__global__ void hyperbolicSine(double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = sinh(a[idx]);
}

__global__ void hyperbolicCosine(double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = cosh(a[idx]);
}

__global__ void hyperbolicTangent(double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    result[idx] = tanh(a[idx]);
}
