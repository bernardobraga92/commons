#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (int i = 5; i * i <= num; i += 6)
        if (num % i == 0 || num % (i + 2) == 0)
            return false;
    return true;
}

__global__ void generatePrimes(int* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        while (!isPrime(idx)) idx++;
        primes[idx] = idx;
    }
}

__device__ void gaussianKernel(float* data, float mean, float stddev, int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        data[i] = mean + stddev * sqrtf(-2.0f * logf(curand_uniform(&globalState)));
    }
}

__global__ void fermatKernel(int* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        long long a = 1 << idx;
        long long b = a + 2;
        while (!isPrime(b)) {
            a += 4;
            b = a + 2;
        }
        primes[idx] = b;
    }
}

__global__ void discConvKernel(int* data, int size) {
    __shared__ int sharedData[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        sharedData[threadIdx.x] = data[idx];

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int tempIdx = (threadIdx.x & ~stride) + (threadIdx.x & stride);
        if (tempIdx < blockDim.x)
            sharedData[threadIdx.x] += sharedData[tempIdx];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        data[blockIdx.x] = sharedData[0];
}

int main() {
    int count = 256;
    int* d_primes;
    cudaMalloc(&d_primes, count * sizeof(int));

    generatePrimes<<<(count + 255) / 256, 256>>>(d_primes, count);
    fermatKernel<<<(count + 255) / 256, 256>>>(d_primes, count);

    int* h_primes = new int[count];
    cudaMemcpy(h_primes, d_primes, count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i)
        printf("%d ", h_primes[i]);
    printf("\n");

    delete[] h_primes;
    cudaFree(d_primes);
    return 0;
}
