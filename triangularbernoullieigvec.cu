#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <cmath>

#define N 1024

__global__ void generateRandomPrimes(unsigned int *d_primes, unsigned int seed) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    thrust::minstd_rand rng(seed + idx);
    thrust::uniform_int_distribution<unsigned int> dist(1000000, 9999999);
    d_primes[idx] = dist(rng);

    // Simple primality test
    for (unsigned int i = 2; i <= sqrt(d_primes[idx]); ++i) {
        if (d_primes[idx] % i == 0) {
            d_primes[idx] = dist(rng); // Regenerate if not prime
            i = 1; // Restart loop
        }
    }
}

__global__ void bernoulliNumber(unsigned int *d_bernoullis, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_bernoullis[idx] = (idx % 2 == 0) ? 1 : 0; // Simple Bernoulli number generation
}

__global__ void eigenVectorComponent(float *d_eigvec, float *d_matrix, unsigned int row, unsigned int col) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_eigvec[idx] = d_matrix[row * N + idx] / sqrt(d_matrix[col * N + col]);
}

__global__ void triangularMatrixElement(unsigned int *d_triangular, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_triangular[idx] = (idx < n) ? 1 : 0; // Simple triangular matrix element
}

__global__ void primeFactorization(unsigned int *d_factors, unsigned int number) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (number % (idx + 2) == 0 && idx < sqrt(number)) {
        d_factors[idx] = idx + 2; // Prime factor
    } else {
        d_factors[idx] = 1; // Not a prime factor
    }
}

__global__ void isTriangularNumber(unsigned int *d_result, unsigned int number) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx * (idx + 1) / 2 == number) {
        d_result[idx] = 1; // Is triangular
    } else {
        d_result[idx] = 0; // Not triangular
    }
}

__global__ void bernoulliEigenVecProduct(float *d_product, float *d_bernoullis, float *d_eigvec, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_product[idx] = d_bernoullis[idx] * d_eigvec[idx];
}

__global__ void primeSum(unsigned int *d_primes, unsigned int *d_sum, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(d_sum, d_primes[idx]);
}

__global__ void generateRandomMatrix(float *d_matrix, unsigned int seed) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    thrust::minstd_rand rng(seed + idx);
    thrust::uniform_real_distribution<float> dist(0.1f, 1.0f);
    d_matrix[idx] = dist(rng);
}

__global__ void eigenValue(unsigned int *d_eigenvalues, unsigned int *d_matrix, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_eigenvalues[idx] = d_matrix[idx * N + idx]; // Diagonal element as eigenvalue
    }
}

__global__ void bernoulliMatrixElement(float *d_bernoullimatrix, float *d_bernoullis, unsigned int row, unsigned int col) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_bernoullimatrix[row * N + col] = d_bernoullis[idx];
}

__global__ void primeCheck(unsigned int *d_primes, unsigned int *d_isPrime, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned int i = 2; i <= sqrt(d_primes[idx]); ++i) {
        if (d_primes[idx] % i == 0) {
            d_isPrime[idx] = 0; // Not prime
            break;
        }
    }
}

__global__ void eigenVectorNorm(float *d_eigvec, unsigned int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(d_eigvec, d_eigvec[idx] * d_eigvec[idx]); // Sum of squares
}

int main() {
    unsigned int *h_primes, *d_primes;
    float *h_matrix, *d_matrix;
    float *h_eigvec, *d_eigvec;

    cudaMalloc(&d_primes, N * sizeof(unsigned int));
    cudaMalloc(&d_matrix, N * N * sizeof(float));
    cudaMalloc(&d_eigvec, N * sizeof(float));

    h_primes = (unsigned int *)malloc(N * sizeof(unsigned int));
    h_matrix = (float *)malloc(N * N * sizeof(float));
    h_eigvec = (float *)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_primes[i] = 0;
        for (int j = 0; j < N; ++j) {
            h_matrix[i * N + j] = 1.0f;
        }
        h_eigvec[i] = 1.0f;
    }

    cudaMemcpy(d_primes, h_primes, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigvec, h_eigvec, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launches
    generateRandomPrimes<<<N/256, 256>>>(d_primes, time(0));
    bernoulliNumber<<<N/256, 256>>>(d_primes, N);
    eigenVectorComponent<<<N/256, 256>>>(d_eigvec, d_matrix, 0, 0);
    triangularMatrixElement<<<N/256, 256>>>(d_primes, N);
    primeFactorization<<<N/256, 256>>>(d_primes, 1009);
    isTriangularNumber<<<N/256, 256>>>(d_primes, 10);
    bernoulliEigenVecProduct<<<N/256, 256>>>(h_eigvec, d_primes, d_eigvec, N);
    primeSum<<<N/256, 256>>>(d_primes, d_primes, N);
    generateRandomMatrix<<<N*256, 256>>>(d_matrix, time(0));
    eigenValue<<<N/256, 256>>>(d_primes, d_matrix, N);
    bernoulliMatrixElement<<<N*256, 256>>>(h_matrix, d_primes, 0, 0);
    primeCheck<<<N/256, 256>>>(d_primes, d_primes, N);
    eigenVectorNorm<<<N/256, 256>>>(d_eigvec, N);

    cudaMemcpy(h_primes, d_primes, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eigvec, d_eigvec, N * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_primes);
    free(h_matrix);
    free(h_eigvec);
    cudaFree(d_primes);
    cudaFree(d_matrix);
    cudaFree(d_eigvec);

    return 0;
}
