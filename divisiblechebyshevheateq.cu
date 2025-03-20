#include <stdio.h>
#include <math.h>

__device__ int is_prime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return 0;
    return 1;
}

__global__ void generate_primes(int* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int num = idx * 2 + 3; // Start from the first odd number after 2
        while (!is_prime(num)) num += 2;
        primes[idx] = num;
    }
}

__device__ bool is_divisible_by_chebyshev(int n, int p) {
    int chebyshev_poly = cos(acos(-1.0) * (n + 0.5) / p);
    return (int)chebyshev_poly % p == 0;
}

__global__ void check_chebyshev_divisibility(int* primes, int count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        for (int i = 2; i <= prime / 2; ++i)
            results[idx] += is_divisible_by_chebyshev(prime, i);
    }
}

__device__ double heat_equation(double x, double y, double t, double alpha) {
    return exp(-alpha * t) * cos(x) * sin(y);
}

__global__ void compute_heat_solution(int* primes, int count, double* solutions, double alpha, double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        solutions[idx] = heat_equation((double)prime / 10.0, (double)prime / 10.0, t, alpha);
    }
}

__global__ void multiply_by_chebyshev(int* primes, int count, double* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        int chebyshev_poly = cos(acos(-1.0) * (prime + 0.5) / prime);
        results[idx] = (double)prime * chebyshev_poly;
    }
}

__global__ void sum_of_squares(int* primes, int count, double* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        results[idx] = (double)prime * prime;
    }
}

__global__ void average_primes(int* primes, int count, double* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double shared_sum[256];
    if (idx < count)
        shared_sum[threadIdx.x] = (double)primes[idx];
    else
        shared_sum[threadIdx.x] = 0.0;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
    }

    if (threadIdx.x == 0)
        atomicAdd(result, shared_sum[0]);
}

__global__ void max_prime(int* primes, int count, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_max[256];
    if (idx < count)
        shared_max[threadIdx.x] = primes[idx];
    else
        shared_max[threadIdx.x] = 0;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s && shared_max[threadIdx.x] < shared_max[threadIdx.x + s])
            shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
    }

    if (threadIdx.x == 0)
        atomicMax(result, shared_max[0]);
}

__global__ void min_prime(int* primes, int count, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_min[256];
    if (idx < count)
        shared_min[threadIdx.x] = primes[idx];
    else
        shared_min[threadIdx.x] = INT_MAX;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s && shared_min[threadIdx.x] > shared_min[threadIdx.x + s])
            shared_min[threadIdx.x] = shared_min[threadIdx.x + s];
    }

    if (threadIdx.x == 0)
        atomicMin(result, shared_min[0]);
}

__global__ void sum_of_divisors(int* primes, int count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        results[idx] = 1; // 1 is a divisor of any number
        for (int i = 2; i <= prime / 2; ++i)
            if (prime % i == 0) results[idx] += i;
    }
}

__global__ void count_divisors(int* primes, int count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int prime = primes[idx];
        results[idx] = 2; // 1 and the number itself are divisors
        for (int i = 2; i <= prime / 2; ++i)
            if (prime % i == 0) results[idx]++;
    }
}

__global__ void product_of_primes(int* primes, int count, unsigned long long* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned long long shared_product[256];
    if (idx < count)
        shared_product[threadIdx.x] = (unsigned long long)primes[idx];
    else
        shared_product[threadIdx.x] = 1;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            shared_product[threadIdx.x] *= shared_product[threadIdx.x + s];
    }

    if (threadIdx.x == 0)
        atomicAdd((unsigned long long*)result, shared_product[0]);
}

__global__ void prime_difference(int* primes, int count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count - 1) {
        results[idx] = primes[idx + 1] - primes[idx];
    }
}

__global__ void reverse_primes(int* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count / 2)
        swap(primes[idx], primes[count - idx - 1]);
}

int main() {
    const int count = 256;
    int* d_primes;
    cudaMalloc(&d_primes, count * sizeof(int));

    generate_primes<<<(count + 255) / 256, 256>>>(d_primes, count);

    int* h_primes = new int[count];
    cudaMemcpy(h_primes, d_primes, count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i)
        printf("%d ", h_primes[i]);
    printf("\n");

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
