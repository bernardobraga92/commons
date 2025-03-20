#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void isPrimeKernel(unsigned long long n, bool* result) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long divisor = 2;

    while (divisor * divisor <= n && !(*result)) {
        if (n % divisor == 0) {
            *result = false;
            break;
        }
        divisor++;
    }

    if (tid == 0) {
        *result = true;
    }
}

bool isPrime(unsigned long long n) {
    bool result;
    bool* d_result;

    cudaMalloc((void**)&d_result, sizeof(bool));
    cudaMemset(d_result, 0, sizeof(bool));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((1 + n - 1) / threadsPerBlock.x);

    isPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_result);

    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return result;
}

__global__ void findNextPrimeKernel(unsigned long long start, unsigned long long* prime) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long candidate = start + tid;

    while (!isPrime(candidate)) {
        candidate++;
    }

    if (tid == 0) {
        *prime = candidate;
    }
}

unsigned long long findNextPrime(unsigned long long start) {
    unsigned long long prime;
    unsigned long long* d_prime;

    cudaMalloc((void**)&d_prime, sizeof(unsigned long long));
    cudaMemset(d_prime, 0, sizeof(unsigned long long));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((1 + (unsigned long long)(sqrt(start) - start)) / threadsPerBlock.x);

    findNextPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, d_prime);

    cudaMemcpy(&prime, d_prime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_prime);

    return prime;
}

__global__ void generatePrimesKernel(unsigned long long start, unsigned long long count, unsigned long long* primes) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        unsigned long long candidate = start + tid;
        while (!isPrime(candidate)) {
            candidate++;
        }
        primes[tid] = candidate;
    }
}

void generatePrimes(unsigned long long start, unsigned long long count, unsigned long long* d_primes) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((count + threadsPerBlock.x - 1) / threadsPerBlock.x);

    generatePrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(start, count, d_primes);
}

__global__ void isPythagoreanTripletKernel(unsigned long long a, unsigned long long b, bool* result) {
    if (a * a + b * b == (a + b) * (a + b)) {
        *result = true;
    }
}

bool isPythagoreanTriplet(unsigned long long a, unsigned long long b) {
    bool result;
    bool* d_result;

    cudaMalloc((void**)&d_result, sizeof(bool));
    cudaMemset(d_result, 0, sizeof(bool));

    isPythagoreanTripletKernel<<<1, 1>>>(a, b, d_result);

    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    return result;
}

__global__ void findNextPythagoreanTripletKernel(unsigned long long startA, unsigned long long* triplet) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = startA + tid;

    for (unsigned long long b = a + 1; !isPythagoreanTriplet(a, b); b++) {}

    if (tid == 0) {
        triplet[0] = a;
        triplet[1] = b;
    }
}

void findNextPythagoreanTriplet(unsigned long long startA, unsigned long long* triplet) {
    unsigned long long* d_triplet;

    cudaMalloc((void**)&d_triplet, sizeof(unsigned long long) * 2);
    cudaMemset(d_triplet, 0, sizeof(unsigned long long) * 2);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((1 + (unsigned long long)(sqrt(startA) - startA)) / threadsPerBlock.x);

    findNextPythagoreanTripletKernel<<<blocksPerGrid, threadsPerBlock>>>(startA, d_triplet);

    cudaMemcpy(triplet, d_triplet, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost);

    cudaFree(d_triplet);
}

__global__ void generatePythagoreanTripletsKernel(unsigned long long startA, unsigned long long count, unsigned long long* triplets) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        unsigned long long a = startA + tid;
        for (unsigned long long b = a + 1; !isPythagoreanTriplet(a, b); b++) {}
        triplets[tid * 2] = a;
        triplets[tid * 2 + 1] = b;
    }
}

void generatePythagoreanTriplets(unsigned long long startA, unsigned long long count, unsigned long long* d_triplets) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((count + threadsPerBlock.x - 1) / threadsPerBlock.x);

    generatePythagoreanTripletsKernel<<<blocksPerGrid, threadsPerBlock>>>(startA, count, d_triplets);
}

__global__ void leastSquaresFitKernel(float* x, float* y, int n, float* coefficients) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_xy = 0.0f;
    float sum_xx = 0.0f;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    __syncthreads();

    atomicAdd(&sum_x, sharedSumX);
    atomicAdd(&sum_y, sharedSumY);
    atomicAdd(&sum_xy, sharedSumXY);
    atomicAdd(&sum_xx, sharedSumXX);

    if (tid == 0) {
        coefficients[0] = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        coefficients[1] = (sum_y - coefficients[0] * sum_x) / n;
    }
}

void leastSquaresFit(float* x, float* y, int n, float* d_coefficients) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    leastSquaresFitKernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, n, d_coefficients);
}
