#include <stdio.h>
#include <stdlib.h>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesKernel(int* primes, int limit, int* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit && isPrime(idx)) {
        atomicAdd(count, 1);
        primes[atomicAdd(count, 0) - 1] = idx;
    }
}

__global__ void jacobiKernel(int* a, int* n, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n[0]) {
        int x = a[idx];
        int y = n[0];
        int res = 1;
        while (x != 0) {
            while (x % 2 == 0) {
                if ((y & 3) == 3) res = -res;
                x /= 2;
            }
            if ((x & 3) == 3 && (y & 3) == 3) res = -res;
            int temp = x;
            x = y % x;
            y = temp;
        }
        result[idx] = y == 1 ? res : 0;
    }
}

__global__ void expKernel(int* base, int* exp, int* mod, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < exp[0]) {
        result[idx] = 1;
        for (int i = 0; i < exp[0]; ++i) {
            result[idx] = (result[idx] * base[idx]) % mod[idx];
        }
    }
}

__global__ void primeJacobiExpKernel(int* primes, int* jacobi, int* exp, int* mod, int limit, int* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < limit) {
        results[idx] = 0;
        if (isPrime(primes[idx])) {
            for (int i = 0; i < jacobi[0]; ++i) {
                if (jacobi[i] == 1) {
                    results[idx] += expKernel<<<1, 1>>>(primes + idx, exp + i, mod + i, results + idx);
                }
            }
        }
    }
}

int main() {
    int limit = 1000;
    int* primes, *count, *jacobi, *exp, *mod, *results;
    cudaMalloc(&primes, limit * sizeof(int));
    cudaMalloc(&count, sizeof(int));
    cudaMalloc(&jacobi, 5 * sizeof(int));
    cudaMalloc(&exp, 5 * sizeof(int));
    cudaMalloc(&mod, 5 * sizeof(int));
    cudaMalloc(&results, limit * sizeof(int));

    findPrimesKernel<<<(limit + 255) / 256, 256>>>(primes, limit, count);

    int* h_primes = (int*)malloc(limit * sizeof(int));
    cudaMemcpy(h_primes, primes, limit * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Primes found: %d\n", *count);

    for (int i = 0; i < 5; ++i) {
        jacobi[i] = rand() % 2 ? -1 : 1;
        exp[i] = rand() % 10 + 1;
        mod[i] = rand() % 100 + 1;
    }

    cudaMemcpy(jacobi, jacobi, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(exp, exp, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mod, mod, 5 * sizeof(int), cudaMemcpyHostToDevice);

    primeJacobiExpKernel<<<(limit + 255) / 256, 256>>>(primes, jacobi, exp, mod, limit, results);

    int* h_results = (int*)malloc(limit * sizeof(int));
    cudaMemcpy(h_results, results, limit * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < limit; ++i) {
        if (h_primes[i] != 0 && h_results[i] != 0) {
            printf("Prime: %d, Result: %d\n", h_primes[i], h_results[i]);
        }
    }

    cudaFree(primes);
    cudaFree(count);
    cudaFree(jacobi);
    cudaFree(exp);
    cudaFree(mod);
    cudaFree(results);

    free(h_primes);
    free(h_results);

    return 0;
}
