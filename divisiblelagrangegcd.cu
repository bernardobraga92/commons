#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void primeCheckKernel(unsigned long long n, bool *isPrime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx <= n / 2) {
        isPrime[idx] = true;
        for (unsigned long long i = 2; i * i <= idx; ++i) {
            if (idx % i == 0) {
                isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void lagrangeInterpolationKernel(int n, double *x, double *y, double xi, double *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double term = y[idx];
        for (int j = 0; j < n; ++j) {
            if (j != idx) {
                term *= (xi - x[j]) / (x[idx] - x[j]);
            }
        }
        atomicAdd(result, term);
    }
}

__global__ void gcdKernel(unsigned long long a, unsigned long long b, unsigned long long *result) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        while (b != 0) {
            unsigned long long temp = b;
            b = a % b;
            a = temp;
        }
        *result = a;
    }
}

__global__ void divisibleKernel(unsigned long long n, unsigned long long divisor, bool *isDivisible) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        isDivisible[idx] = (idx % divisor == 0);
    }
}

extern "C" void primeCheck(unsigned long long n, bool *d_isPrime, cudaStream_t stream) {
    primeCheckKernel<<<(n / 2 + 255) / 256, 256, 0, stream>>>(n, d_isPrime);
}

extern "C" void lagrangeInterpolation(int n, double *x, double *y, double xi, double *result, cudaStream_t stream) {
    lagrangeInterpolationKernel<<<(n + 255) / 256, 256, 0, stream>>>(n, x, y, xi, result);
}

extern "C" void gcd(unsigned long long a, unsigned long long b, unsigned long long *result, cudaStream_t stream) {
    gcdKernel<<<1, 1, 0, stream>>>(a, b, result);
}

extern "C" void divisible(unsigned long long n, unsigned long long divisor, bool *d_isDivisible, cudaStream_t stream) {
    divisibleKernel<<<(n + 255) / 256, 256, 0, stream>>>(n, divisor, d_isDivisible);
}
