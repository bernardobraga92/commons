#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256

__global__ void harmonicKernel(float *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = 1.0f / (idx + 1);
    }
}

__global__ void lagrangeKernel(float *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = pow(-1.0f, idx);
    }
}

__global__ void vecAddKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void harmonicLagrangeVecAdd(int size) {
    float *h_arrayA = new float[size];
    float *h_arrayB = new float[size];
    float *h_arrayC = new float[size];

    for (int i = 0; i < size; ++i) {
        h_arrayA[i] = 0.0f;
        h_arrayB[i] = 0.0f;
        h_arrayC[i] = 0.0f;
    }

    float *d_arrayA, *d_arrayB, *d_arrayC;

    cudaMalloc(&d_arrayA, size * sizeof(float));
    cudaMalloc(&d_arrayB, size * sizeof(float));
    cudaMalloc(&d_arrayC, size * sizeof(float));

    harmonicKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_arrayA, size);
    lagrangeKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_arrayB, size);
    vecAddKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_arrayA, d_arrayB, d_arrayC, size);

    cudaMemcpy(h_arrayC, d_arrayC, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        std::cout << h_arrayC[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_arrayA);
    cudaFree(d_arrayB);
    cudaFree(d_arrayC);

    delete[] h_arrayA;
    delete[] h_arrayB;
    delete[] h_arrayC;
}

int main() {
    harmonicLagrangeVecAdd(1024);
    return 0;
}
