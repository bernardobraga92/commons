#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (unsigned long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void generatePrimes(unsigned long long *primes, unsigned int count, curandState *state) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        curand_init(clock64(), idx, 0, &state[idx]);
        unsigned long long candidate = curand(&state[idx]) % 1000000007;
        while (!isPrime(candidate)) {
            candidate = curand(&state[idx]) % 1000000007;
        }
        primes[idx] = candidate;
    }
}

__global__ void addLargePrimes(unsigned long long *primes, unsigned int count) {
    __shared__ unsigned long long sharedPrimes[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        sharedPrimes[threadIdx.x] = primes[idx];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedPrimes[threadIdx.x] += sharedPrimes[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        primes[blockIdx.x] = sharedPrimes[0];
    }
}

int main() {
    const unsigned int count = 1024;
    unsigned long long *h_primes, *d_primes;
    curandState *d_state;

    h_primes = new unsigned long long[count];
    cudaMalloc(&d_primes, count * sizeof(unsigned long long));
    cudaMalloc(&d_state, count * sizeof(curandState));

    generatePrimes<<<(count + 255) / 256, 256>>>(d_primes, count, d_state);
    addLargePrimes<<<(count + 255) / 256, 256>>>(d_primes, count);

    cudaMemcpy(h_primes, d_primes, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < count; ++i) {
        printf("%llu\n", h_primes[i]);
    }

    delete[] h_primes;
    cudaFree(d_primes);
    cudaFree(d_state);

    return 0;
}
