#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void eulerCheckKernel1(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void eulerCheckKernel2(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 3 != 0) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void eulerCheckKernel3(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 10) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void eulerCheckKernel4(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] % 2 != 0) {
        results[idx] = isPrime(numbers[idx]);
    }
}

// ... Repeat for a total of 40 kernels ...

__global__ void eulerCheckKernel40(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && numbers[idx] > 100) {
        results[idx] = isPrime(numbers[idx]);
    }
}

int main() {
    const int size = 1024;
    int h_numbers[size];
    bool h_results[size];
    for (int i = 0; i < size; ++i) {
        h_numbers[i] = rand() % 500 + 2;
    }

    int* d_numbers;
    bool* d_results;
    cudaMalloc(&d_numbers, size * sizeof(int));
    cudaMalloc(&d_results, size * sizeof(bool));

    cudaMemcpy(d_numbers, h_numbers, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    eulerCheckKernel1<<<gridSize, blockSize>>>(d_numbers, size, d_results);
    // ... Call other kernels ...

    eulerCheckKernel40<<<gridSize, blockSize>>>(d_numbers, size, d_results);

    cudaMemcpy(h_results, d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (h_results[i]) {
            std::cout << h_numbers[i] << " is prime." << std::endl;
        }
    }

    cudaFree(d_numbers);
    cudaFree(d_results);

    return 0;
}
