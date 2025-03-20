#include <cuda_runtime.h>
#include <cmath>

__device__ bool is_prime(int n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= sqrt(n); i += 2)
        if (n % i == 0) return false;
    return true;
}

__global__ void find_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_prime(idx)) primes[idx] = idx;
}

__device__ int perfect_weyl_mean(int a, int b, int c) {
    return (a + b + c) / 3;
}

__global__ void weyl_mean_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_prime(idx))
        primes[idx] = perfect_weyl_mean(idx, idx * idx, idx * idx * idx);
}

__device__ bool is_weyl_prime(int n) {
    return is_prime(n) && is_prime(n * n) && is_prime(n * n * n);
}

__global__ void find_weyl_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_weyl_prime(idx)) primes[idx] = idx;
}

__device__ int weyl_sum(int a, int b, int c) {
    return a + b + c;
}

__global__ void sum_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_prime(idx))
        primes[idx] = weyl_sum(idx, idx * idx, idx * idx * idx);
}

__device__ bool is_even_prime(int n) {
    return n == 2;
}

__global__ void find_even_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_even_prime(idx)) primes[idx] = idx;
}

__device__ bool is_odd_prime(int n) {
    return is_prime(n) && n % 2 != 0;
}

__global__ void find_odd_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_odd_prime(idx)) primes[idx] = idx;
}

__device__ bool is_superprime(int n) {
    return is_prime(n) && is_prime(prime_count(n));
}

__global__ void find_superprimes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_superprime(idx)) primes[idx] = idx;
}

__device__ bool is_twin_prime(int n) {
    return is_prime(n) && (is_prime(n - 2) || is_prime(n + 2));
}

__global__ void find_twin_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_twin_prime(idx)) primes[idx] = idx;
}

__device__ bool is_cousin_prime(int n) {
    return is_prime(n) && (is_prime(n - 4) || is_prime(n + 4));
}

__global__ void find_cousin_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_cousin_prime(idx)) primes[idx] = idx;
}

__device__ bool is_sexty_prime(int n) {
    return is_prime(n) && (is_prime(n - 60) || is_prime(n + 60));
}

__global__ void find_sexty_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_sexty_prime(idx)) primes[idx] = idx;
}

__device__ bool is_amicable_pair(int a, int b) {
    return sum_proper_divisors(a) == b && sum_proper_divisors(b) == a;
}

__global__ void find_amicable_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_prime(idx))
        primes[idx] = sum_proper_divisors(idx);
}

__device__ bool is_perfect_number(int n) {
    return sum_proper_divisors(n) == n;
}

__global__ void find_perfect_primes(int *primes, int max, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && is_perfect_number(idx)) primes[idx] = idx;
}

int main() {
    const int size = 10000;
    int h_primes[size];
    int *d_primes;

    cudaMalloc((void **)&d_primes, size * sizeof(int));

    find_primes<<<256, 256>>>(d_primes, size, size);

    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_primes);

    return 0;
}
