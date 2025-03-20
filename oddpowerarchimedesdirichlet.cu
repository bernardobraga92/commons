#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_BLOCK_SIZE 256

__device__ bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void odd_power_archimedes_dirichlet(int *primes, int limit, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * 2 + 1; // Generate only odd numbers
    if (num <= limit && is_prime(num)) {
        atomicAdd(count, 1);
        primes[atomicSub(count, 1)] = num;
    }
}

int main() {
    const int limit = 1000000;
    const int max_primes = limit / 2; // Estimate for odd numbers
    int *primes_h = (int *)malloc(max_primes * sizeof(int));
    int *primes_d, *count_d;
    int count = 0;

    cudaMalloc(&primes_d, max_primes * sizeof(int));
    cudaMalloc(&count_d, sizeof(int));

    cudaMemset(count_d, 0, sizeof(int));

    odd_power_archimedes_dirichlet<<<(limit + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(primes_d, limit, count_d);
    cudaMemcpyFromSymbol(&count, count_d, sizeof(int));
    cudaMemcpy(primes_h, primes_d, count * sizeof(int));

    printf("Number of primes found: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("%d ", primes_h[i]);
    }
    printf("\n");

    cudaFree(primes_d);
    cudaFree(count_d);
    free(primes_h);

    return 0;
}
