#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define BLOCK_SIZE 256

__global__ void isPrimeKernel(unsigned long *candidates, bool *results, int numCandidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCandidates) return;

    unsigned long candidate = candidates[idx];
    results[idx] = true;

    for (unsigned long i = 2; i <= sqrt(candidate); ++i) {
        if (candidate % i == 0) {
            results[idx] = false;
            break;
        }
    }
}

__global__ void generateCandidatesKernel(unsigned long *candidates, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (end - start)) return;

    candidates[idx] = start + idx * 2;
}

__global__ void cauchyMeanKernel(double *data, double *result, int n) {
    __shared__ double sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sharedData[tid] = data[i];
    } else {
        sharedData[tid] = 0.0;
    }

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

__global__ void evenCauchyMeanKernel(double *data, double *result, int n) {
    __shared__ double sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && i % 2 == 0) {
        sharedData[tid] = data[i];
    } else {
        sharedData[tid] = 0.0;
    }

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

__global__ void sumKernel(double *data, double *result, int n) {
    __shared__ double sharedSum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sharedSum[tid] = data[i];
    } else {
        sharedSum[tid] = 0.0;
    }

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedSum[0];
    }
}

__global__ void multiplyKernel(double *data, double *result, int n) {
    __shared__ double sharedProduct[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sharedProduct[tid] = data[i];
    } else {
        sharedProduct[tid] = 1.0;
    }

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedProduct[tid] *= sharedProduct[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedProduct[0];
    }
}

__global__ void normalizeKernel(double *data, double mean, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] -= mean;
        data[idx] /= sqrt(mean);
    }
}

int main() {
    const int numCandidates = 1024;
    unsigned long *candidates, *d_candidates;
    bool *results, *d_results;

    cudaMalloc(&d_candidates, numCandidates * sizeof(unsigned long));
    cudaMalloc(&d_results, numCandidates * sizeof(bool));

    generateCandidatesKernel<<<(numCandidates + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_candidates, 2, numCandidates * 2);
    isPrimeKernel<<<(numCandidates + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_candidates, d_results, numCandidates);

    cudaMemcpy(candidates, d_candidates, numCandidates * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    cudaMemcpy(results, d_results, numCandidates * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numCandidates; ++i) {
        if (results[i]) {
            std::cout << candidates[i] << " is prime." << std::endl;
        }
    }

    cudaFree(d_candidates);
    cudaFree(d_results);

    return 0;
}
