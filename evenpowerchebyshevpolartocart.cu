#include <math.h>

__global__ void chebyshevPolynomialTransform(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = cosf(acosf(x[idx]) * (2.0f * idx + 1) / (4.0f * n));
    }
}

__global__ void powerTransform(float *x, int n, int power) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        x[idx] = powf(x[idx], power);
    }
}

__global__ void primeCheck(int *numbers, bool *isPrime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && numbers[idx] > 1) {
        isPrime[idx] = true;
        for (int i = 2; i <= sqrtf(numbers[idx]); ++i) {
            if (numbers[idx] % i == 0) {
                isPrime[idx] = false;
                break;
            }
        }
    } else {
        isPrime[idx] = false;
    }
}

__global__ void generateRandomPrimes(int *primes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        primes[idx] = rand() % 100000 + 2; // Random number between 2 and 99999
        primeCheck<<<(n + 255) / 256, 256>>>(primes, &primes[idx], n);
    }
}

__global__ void evenPowerChebyshevPolarToCart(float *r, float *theta, float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        x[idx] = r[idx] * cosf(theta[idx]);
        y[idx] = r[idx] * sinf(theta[idx]);
        powerTransform<<<1, 256>>>(x, n, 2);
        chebyshevPolynomialTransform<<<(n + 255) / 256, 256>>>(x, y, n);
    }
}

int main() {
    const int n = 1024;
    float *d_x, *d_y, *d_r, *d_theta;
    bool *d_isPrime;
    int *h_numbers, *h_primes;

    h_numbers = new int[n];
    h_primes = new int[n];

    for (int i = 0; i < n; ++i) {
        h_numbers[i] = rand() % 10000 + 2;
    }

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_r, n * sizeof(float));
    cudaMalloc((void **)&d_theta, n * sizeof(float));
    cudaMalloc((void **)&d_isPrime, n * sizeof(bool));

    cudaMemcpy(d_x, h_numbers, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_numbers, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_numbers, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, h_numbers, n * sizeof(float), cudaMemcpyHostToDevice);

    primeCheck<<<(n + 255) / 256, 256>>>(d_x, d_isPrime, n);

    evenPowerChebyshevPolarToCart<<<(n + 255) / 256, 256>>>(d_r, d_theta, d_x, d_y, n);
    generateRandomPrimes<<<(n + 255) / 256, 256>>>(h_primes, n);

    cudaMemcpy(h_numbers, d_isPrime, n * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_primes, h_primes, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r);
    cudaFree(d_theta);
    cudaFree(d_isPrime);

    delete[] h_numbers;
    delete[] h_primes;

    return 0;
}
