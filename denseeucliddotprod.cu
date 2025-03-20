#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void findLargePrimes(int *primes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    int n = idx * 2 + 3; // Generate odd numbers starting from 3
    bool isPrime = true;

    for (int i = 2; i * i <= n && isPrime; ++i)
        if (n % i == 0) isPrime = false;

    primes[idx] = isPrime ? n : 0;
}

__global__ void euclideanDistance(int *a, int *b, float *distances, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    float dx = a[idx * 2] - b[idx * 2];
    float dy = a[idx * 2 + 1] - b[idx * 2 + 1];
    distances[idx] = sqrtf(dx * dx + dy * dy);
}

__global__ void dotProduct(float *a, float *b, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    results[idx] = a[idx] * b[idx];
}

__global__ void matrixTranspose(int *matrix, int rows, int cols, int *transposed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= cols || idy >= rows) return;

    transposed[idy * cols + idx] = matrix[idx * rows + idy];
}

__global__ void vectorAddition(float *a, float *b, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    results[idx] = a[idx] + b[idx];
}

__global__ void matrixMultiplication(int *A, int *B, int *C, int M, int N, int P) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= P) return;

    C[row * P + col] = 0;
    for (int k = 0; k < N; ++k)
        C[row * P + col] += A[row * N + k] * B[k * P + col];
}

__global__ void vectorMultiplication(float *a, float *b, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    results[idx] = a[idx] * b[idx];
}

__global__ void matrixInverse(int *matrix, int size, int *inverse) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= size || col >= size) return;

    inverse[row * size + col] = (row == col) ? 1 : 0;
}

__global__ void scalarMultiplication(float *vector, float scalar, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    results[idx] = vector[idx] * scalar;
}

__global__ void matrixDeterminant(int *matrix, int size, int *determinant) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= size || col >= size) return;

    determinant[row] = (row == col) ? matrix[row * size + col] : 0;
}

__global__ void vectorNorm(float *vector, float *norms, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    norms[idx] = sqrtf(vector[idx] * vector[idx]);
}

__global__ void matrixAddition(int *A, int *B, int *C, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= N) return;

    C[row * N + col] = A[row * N + col] + B[row * N + col];
}

__global__ void vectorSubtraction(float *a, float *b, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    results[idx] = a[idx] - b[idx];
}

__global__ void matrixSubtraction(int *A, int *B, int *C, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= N) return;

    C[row * N + col] = A[row * N + col] - B[row * N + col];
}

__global__ void matrixElementwiseMultiplication(int *A, int *B, int *C, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= N) return;

    C[row * N + col] = A[row * N + col] * B[row * N + col];
}

__global__ void vectorDivide(float *a, float *b, float *results, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count || b[idx] == 0) return;

    results[idx] = a[idx] / b[idx];
}

__global__ void matrixElementwiseDivision(int *A, int *B, int *C, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= N || B[row * N + col] == 0) return;

    C[row * N + col] = A[row * N + col] / B[row * N + col];
}

__global__ void vectorMagnitude(float *vector, float *magnitudes, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    magnitudes[idx] = sqrtf(vector[idx] * vector[idx]);
}

__global__ void matrixTrace(int *matrix, int size, int *trace) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= size || col >= size) return;

    trace[row] = (row == col) ? matrix[row * size + col] : 0;
}

__global__ void vectorCrossProduct(float *a, float *b, float *cross) {
    int idx = threadIdx.x;
    if (idx >= 3) return;

    cross[idx] = a[(idx + 1) % 3] * b[(idx + 2) % 3] - a[(idx + 2) % 3] * b[(idx + 1) % 3];
}

__global__ void matrixCofactor(int *matrix, int size, int *cofactor, int row, int col) {
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= size || c >= size) return;

    if (r < row && c < col)
        cofactor[r * (size - 1) + c] = matrix[r * size + c];
    else if (r < row && c > col)
        cofactor[r * (size - 1) + c - 1] = matrix[r * size + c];
    else if (r > row && c < col)
        cofactor[(r - 1) * (size - 1) + c] = matrix[r * size + c];
    else if (r > row && c > col)
        cofactor[(r - 1) * (size - 1) + c - 1] = matrix[r * size + c];
}

__global__ void vectorDotProduct(float *a, float *b, float *result, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    result[0] += a[idx] * b[idx];
}
