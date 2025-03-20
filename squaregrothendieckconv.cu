#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ int isPrime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void findPrimesKernel(int *d_primes, int max_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_num && isPrime(idx)) {
        d_primes[idx] = 1;
    } else {
        d_primes[idx] = 0;
    }
}

void findPrimes(int *h_primes, int size) {
    int *d_primes;
    cudaMalloc((void **)&d_primes, size * sizeof(int));
    memset(h_primes, 0, size * sizeof(int));

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    findPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);

    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

__global__ void squareKernel(int *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] *= d_array[idx];
    }
}

void squareArray(int *h_array, int size) {
    int *d_array;
    cudaMalloc((void **)&d_array, size * sizeof(int));
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, size);

    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

__global__ void grothendieckKernel(float *d_values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_values[idx] = sinf(d_values[idx]) + cosf(d_values[idx]);
    }
}

void grothendieckTransform(float *h_values, int size) {
    float *d_values;
    cudaMalloc((void **)&d_values, size * sizeof(float));
    cudaMemcpy(d_values, h_values, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    grothendieckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, size);

    cudaMemcpy(h_values, d_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_values);
}

__global__ void convolveKernel(int *d_data, int *d_kernel, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int k = 0; k < size; k++) {
            d_data[idx] += d_data[k] * d_kernel[idx - k];
        }
    }
}

void convolve(int *h_data, int *h_kernel, int size) {
    int *d_data, *d_kernel;
    cudaMalloc((void **)&d_data, size * sizeof(int));
    cudaMalloc((void **)&d_kernel, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_kernel, size);

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_kernel);
}
