#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void generateRandomPrimes(unsigned int *primes, unsigned int num_primes, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    curand_init(clock64(), idx, 0, &state[idx]);
    unsigned int candidate;
    do {
        candidate = curand(&state[idx]) % 1000000 + 2; // Random number between 2 and 999999
    } while (!isPrime(candidate));

    primes[idx] = candidate;
}

__device__ bool isPrime(unsigned int num) {
    if (num < 2) return false;
    for (unsigned int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void sigmoidTransform(unsigned int *primes, float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = static_cast<float>(primes[idx]) / 1000000.0f;
    sigmoid_primes[idx] = 1.0f / (1.0f + expf(-x));
}

__global__ void euclideanDistance(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes - 1) return;

    float dx = sigmoid_primes[idx] - sigmoid_primes[idx + 1];
    float dy = 0.0f; // Assuming y values are constant
    sigmoid_primes[idx] = sqrtf(dx * dx + dy * dy);
}

__global__ void reluTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    sigmoid_primes[idx] = max(0.0f, sigmoid_primes[idx]);
}

__global__ void tanhTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    sigmoid_primes[idx] = tanhf(sigmoid_primes[idx]);
}

__global__ void softplusTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    sigmoid_primes[idx] = logf(1.0f + expf(sigmoid_primes[idx]));
}

__global__ void eluTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    sigmoid_primes[idx] = sigmoid_primes[idx] > 0 ? sigmoid_primes[idx] : expf(sigmoid_primes[idx]) - 1.0f;
}

__global__ void leakyreluTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    sigmoid_primes[idx] = sigmoid_primes[idx] > 0 ? sigmoid_primes[idx] : 0.1f * sigmoid_primes[idx];
}

__global__ void mishTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x * tanhf(logf(1.0f + expf(x)));
}

__global__ void hardswishTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x * fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
}

__global__ void seluTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x > 0 ? 1.0507f * x : 1.0507f * (expf(x) - 1.0f);
}

__global__ void swishTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x * sigmoidf(x);
}

__global__ void celuTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float alpha = 1.0f;
    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
}

__global__ void gelfTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = 0.5f * (expf(x) - expf(-x));
}

__global__ void bentIdentityTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x + ((sqrtf(x * x + 1.0f) - 1.0f) / 2.0f);
}

__global__ void erfTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = erff(x);
}

__global__ void softsignTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float x = sigmoid_primes[idx];
    sigmoid_primes[idx] = x / (1.0f + abs(x));
}

__global__ void softminTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float min_val = sigmoid_primes[idx];
    for (int i = 0; i < num_primes; ++i) {
        min_val = fminf(min_val, sigmoid_primes[i]);
    }
    sigmoid_primes[idx] = expf(-sigmoid_primes[idx]) / expf(-min_val);
}

__global__ void softmaxTransform(float *sigmoid_primes, unsigned int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primes) return;

    float sum_exp = 0.0f;
    for (int i = 0; i < num_primes; ++i) {
        sum_exp += expf(sigmoid_primes[i]);
    }
    sigmoid_primes[idx] = expf(sigmoid_primes[idx]) / sum_exp;
}
