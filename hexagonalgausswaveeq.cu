#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void findLargePrimesKernel(unsigned long long *d_primes, unsigned int numPrimes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        // Generate a random number
        unsigned long long candidate = rand() % 1000000000000ULL + 2;

        // Check for primality using trial division
        bool isPrime = true;
        for (unsigned long long i = 2; i <= sqrt(candidate); ++i) {
            if (candidate % i == 0) {
                isPrime = false;
                break;
            }
        }

        if (isPrime) {
            d_primes[idx] = candidate;
        } else {
            d_primes[idx] = 0; // Mark non-prime
        }
    }
}

unsigned long long *findLargePrimes(unsigned int numPrimes, unsigned int blockSize, unsigned int gridSize) {
    unsigned long long *h_primes = (unsigned long long *)malloc(numPrimes * sizeof(unsigned long long));
    unsigned long long *d_primes;
    cudaMalloc(&d_primes, numPrimes * sizeof(unsigned long long));

    findLargePrimesKernel<<<gridSize, blockSize>>>(d_primes, numPrimes);

    cudaMemcpy(h_primes, d_primes, numPrimes * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_primes);
    return h_primes;
}

__global__ void generateHexagonalNumbersKernel(unsigned long long *d_hexagons, unsigned int numHexagons) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numHexagons) {
        d_hexagons[idx] = 3 * idx * idx - idx + 1; // nth hexagonal number
    }
}

unsigned long long *generateHexagonalNumbers(unsigned int numHexagons, unsigned int blockSize, unsigned int gridSize) {
    unsigned long long *h_hexagons = (unsigned long long *)malloc(numHexagons * sizeof(unsigned long long));
    unsigned long long *d_hexagons;
    cudaMalloc(&d_hexagons, numHexagons * sizeof(unsigned long long));

    generateHexagonalNumbersKernel<<<gridSize, blockSize>>>(d_hexagons, numHexagons);

    cudaMemcpy(h_hexagons, d_hexagons, numHexagons * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_hexagons);
    return h_hexagons;
}

__global__ void applyGaussianWaveEquationKernel(float *d_wave, unsigned int gridSize) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gridSize) {
        float x = idx - gridSize / 2.0f;
        d_wave[idx] = exp(-x * x / 100.0f) * sin(x);
    }
}

void applyGaussianWaveEquation(float *h_wave, unsigned int gridSize, unsigned int blockSize, unsigned int gridSizeKernel) {
    float *d_wave;
    cudaMalloc(&d_wave, gridSize * sizeof(float));

    applyGaussianWaveEquationKernel<<<gridSizeKernel, blockSize>>>(d_wave, gridSize);

    cudaMemcpy(h_wave, d_wave, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_wave);
}

__global__ void complexFunction1(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = d_input[idx] * d_input[idx] + 3 * d_input[idx] - 7;
    }
}

__global__ void complexFunction2(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = sqrt(d_input[idx]) + log(d_input[idx]);
    }
}

__global__ void complexFunction3(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = pow(d_input[idx], 2) - pow(d_input[idx], 3) + 5;
    }
}

__global__ void complexFunction4(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = sin(d_input[idx]) + cos(d_input[idx]);
    }
}

__global__ void complexFunction5(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = tanh(d_input[idx]) * sinh(d_input[idx]);
    }
}

__global__ void complexFunction6(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = exp(-d_input[idx]) * cos(d_input[idx]);
    }
}

__global__ void complexFunction7(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = sqrt(abs(d_input[idx])) * log10(d_input[idx]);
    }
}

__global__ void complexFunction8(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = pow(d_input[idx], 4) - pow(d_input[idx], 2) + 9;
    }
}

__global__ void complexFunction9(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = asin(d_input[idx]) + acos(d_input[idx]);
    }
}

__global__ void complexFunction10(unsigned long long *d_input, unsigned long long *d_output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = atan(d_input[idx]) * exp(d_input[idx]);
    }
}

int main() {
    const unsigned int numPrimes = 40;
    const unsigned int blockSize = 256;
    const unsigned int gridSize = (numPrimes + blockSize - 1) / blockSize;

    unsigned long long *h_primes = findLargePrimes(numPrimes, blockSize, gridSize);

    for (unsigned int i = 0; i < numPrimes; ++i) {
        if (h_primes[i] != 0) {
            printf("Prime: %llu\n", h_primes[i]);
        }
    }

    free(h_primes);
    return 0;
}
