#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <iostream>

#define N 1024

__global__ void findPrimes(int* d_primes, int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    bool isPrime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            isPrime = false;
            break;
        }
    }

    d_primes[idx] = isPrime ? idx : 0;
}

__global__ void multiplyMatrices(int* A, int* B, int* C, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void generateRandomPrimes(int* d_primes, int limit) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;

    curandState state;
    curand_init(clock64(), idx, 0, &state);
    int num = curand(&state) % (limit - 2) + 2;
    bool isPrime = true;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) {
            isPrime = false;
            break;
        }
    }

    d_primes[idx] = isPrime ? num : 0;
}

__global__ void addMatrices(int* A, int* B, int* C, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
        C[row * N + col] = A[row * N + col] + B[row * N + col];
}

__global__ void subtractMatrices(int* A, int* B, int* C, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
        C[row * N + col] = A[row * N + col] - B[row * N + col];
}

__global__ void transposeMatrix(int* A, int* AT, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
        AT[col * N + row] = A[row * N + col];
}

__global__ void computeDeterminant(int* A, float* det) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    float a = A[0], b = A[1], c = A[2], d = A[3];
    *det = a * d - b * c;
}

__global__ void inverseMatrix(int* A, int* AI, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    float det = A[0] * A[3] - A[1] * A[2];
    if (det == 0) return;

    AI[0] = static_cast<int>(A[3] / det);
    AI[1] = static_cast<int>(-A[1] / det);
    AI[2] = static_cast<int>(-A[2] / det);
    AI[3] = static_cast<int>(A[0] / det);
}

__global__ void findMaxInRow(int* A, int* maxVal, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;

    maxVal[row] = A[row * N];
    for (int col = 1; col < N; ++col) {
        if (A[row * N + col] > maxVal[row])
            maxVal[row] = A[row * N + col];
    }
}

__global__ void findMinInColumn(int* A, int* minVal, int N) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    minVal[col] = A[col];
    for (int row = 1; row < N; ++row) {
        if (A[row * N + col] < minVal[col])
            minVal[col] = A[row * N + col];
    }
}

__global__ void scaleMatrix(int* A, int scalar, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
        A[row * N + col] *= scalar;
}

__global__ void computeTrace(int* A, int* trace, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    int sum = 0;
    for (int i = 0; i < N; ++i)
        sum += A[i * N + i];
    *trace = sum;
}

__global__ void isSymmetric(int* A, bool* result, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    for (int row = 0; row < N; ++row)
        for (int col = row + 1; col < N; ++col)
            if (A[row * N + col] != A[col * N + row]) {
                *result = false;
                return;
            }
    *result = true;
}

__global__ void isDiagonal(int* A, bool* result, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            if (row != col && A[row * N + col] != 0) {
                *result = false;
                return;
            }
    *result = true;
}

__global__ void isIdentity(int* A, bool* result, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            if ((row == col && A[row * N + col] != 1) || (row != col && A[row * N + col] != 0)) {
                *result = false;
                return;
            }
    *result = true;
}

__global__ void computeEigenvalues(int* A, float* eigenvals, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    // Placeholder for eigenvalue computation
    eigenvals[0] = 1.0f; // Example value
    eigenvals[1] = 2.0f; // Example value
}

__global__ void computeEigenvectors(int* A, float* eigenvects, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    // Placeholder for eigenvector computation
    eigenvects[0] = 1.0f; // Example value
    eigenvects[1] = 2.0f; // Example value
}

int main() {
    int* d_primes;
    cudaMalloc(&d_primes, N * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Example usage of some functions
    findPrimes<<<numBlocks, threadsPerBlock>>>(d_primes, N);
    generateRandomPrimes<<<numBlocks, threadsPerBlock>>>(d_primes, N);

    int* h_primes = new int[N];
    cudaMemcpy(h_primes, d_primes, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d ", h_primes[i]);
    }
    printf("\n");

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
