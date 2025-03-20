#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

__global__ void is_prime_kernel(unsigned int *primes, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    primes[idx] = 1;
    for (unsigned int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void generate_random_primes(unsigned int *primes, unsigned int limit, curandState *state) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    curand_init(clock() + idx, 0, 0, &state[idx]);
    unsigned int num = curand(&state[idx]) % limit;
    primes[idx] = (is_prime(num)) ? num : 1;
}

__global__ void stddev_kernel(unsigned int *primes, float *stddev, unsigned int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    atomicAdd(stddev, primes[idx]);
    atomicAdd(&stddev[1], primes[idx] * primes[idx]);
}

__global__ void average_kernel(float *stddev, unsigned int limit) {
    float sum = stddev[0];
    float sum_sq = stddev[1];
    stddev[0] = sum / limit;
    stddev[1] = sqrt((sum_sq / limit) - (stddev[0] * stddev[0]));
}

extern "C" void algebraiceratosthenesstddev(unsigned int *primes, unsigned int limit, float *stddev) {
    curandState *state;
    cudaMalloc(&state, limit * sizeof(curandState));

    is_prime_kernel<<<(limit + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(primes, limit);
    generate_random_primes<<<(limit + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(primes, limit, state);
    stddev_kernel<<<(limit + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(primes, stddev, limit);
    average_kernel<<<1, 1>>>(stddev, limit);

    cudaFree(state);
}
