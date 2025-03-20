#include <iostream>
#include <cuda_runtime.h>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* d_primes, int numPrimes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPrimes) {
        d_primes[idx] = (isPrime(idx) ? idx : 0);
    }
}

__global__ void multiplyAndActivate(int* d_array, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] *= static_cast<int>(factor * sin(d_array[idx]) + cos(d_array[idx]));
    }
}

__global__ void gaussianActivation(int* d_array, int size, float mean, float stdDev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(exp(-0.5f * pow((d_array[idx] - mean) / stdDev, 2)) * 100);
    }
}

__global__ void addGaussianNoise(int* d_array, int size, float mean, float stdDev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] += static_cast<int>(mean + stdDev * sin(d_array[idx]));
    }
}

__global__ void reluActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = (d_array[idx] > 0 ? d_array[idx] : 0);
    }
}

__global__ void leakyReluActivation(int* d_array, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = (d_array[idx] > 0 ? d_array[idx] : alpha * d_array[idx]);
    }
}

__global__ void sigmoidActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(1.0f / (1.0f + exp(-d_array[idx])));
    }
}

__global__ void softmaxActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += exp(d_array[i]);
    }
    if (idx < size) {
        d_array[idx] = static_cast<int>(exp(d_array[idx]) / sum);
    }
}

__global__ void tanhActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(tanh(d_array[idx]));
    }
}

__global__ void eluActivation(int* d_array, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = (d_array[idx] > 0 ? d_array[idx] : alpha * (exp(d_array[idx]) - 1));
    }
}

__global__ void seluActivation(int* d_array, int size, float lambda, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = (d_array[idx] > 0 ? lambda * d_array[idx] : lambda * alpha * (exp(d_array[idx]) - 1));
    }
}

__global__ void swishActivation(int* d_array, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(d_array[idx] / (1.0f + exp(-beta * d_array[idx])));
    }
}

__global__ void mishActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = tanh(softplus(d_array[idx]));
        d_array[idx] = static_cast<int>(d_array[idx] * y);
    }
}

__global__ void softplusActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(log(1.0f + exp(d_array[idx])));
    }
}

__global__ void hardSwishActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = d_array[idx];
        d_array[idx] = static_cast<int>(x * max(0.0f, min(1.0f, (x + 3.0f) / 6.0f)));
    }
}

__global__ void hardTanhActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(max(-1.0f, min(1.0f, d_array[idx])));
    }
}

__global__ void thresholdedReluActivation(int* d_array, int size, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = (d_array[idx] > theta ? d_array[idx] : 0);
    }
}

__global__ void exponentialActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = static_cast<int>(exp(d_array[idx]));
    }
}

__global__ void linearActivation(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] *= 2; // Example linear transformation
    }
}
