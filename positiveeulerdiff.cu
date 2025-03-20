#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

__device__ bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void find_primes(int* d_numbers, int size, bool* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_results[idx] = is_prime(d_numbers[idx]);
    }
}

void generate_large_primes(int* h_numbers, int size) {
    thrust::host_vector<int> h_vec(size);
    std::generate(h_vec.begin(), h_vec.end(), []() { return rand() % 1000000; });
    thrust::copy(h_vec.begin(), h_vec.end(), h_numbers);
}

void check_primes(int* h_results, int size) {
    for (int i = 0; i < size; ++i) {
        if (!h_results[i]) {
            printf("Not a prime: %d\n", h_results[i]);
        }
    }
}

__global__ void euler_totient_function(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = d_numbers[idx];
        int result = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                result -= result / i;
            }
        }
        if (num > 1) result -= result / num;
        d_results[idx] = result;
    }
}

__global__ void euler_totient_diff(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    }
}

__global__ void euler_totient_diff_prime(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(d_numbers[idx])) {
        int num = d_numbers[idx];
        int totient = num - 1;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_composite(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !is_prime(d_numbers[idx])) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_large(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] > 10000) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_small(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] <= 10000) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_odd(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] % 2 != 0) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_even(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && d_numbers[idx] % 2 == 0) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_coprime(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; ++j) {
            if (gcd(d_numbers[idx], d_numbers[j]) == 1) {
                d_results[idx] += euler_totient_function(d_numbers[j]);
            }
        }
    }
}

__global__ void euler_totient_diff_non_coprime(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; ++j) {
            if (gcd(d_numbers[idx], d_numbers[j]) != 1) {
                d_results[idx] += euler_totient_function(d_numbers[j]);
            }
        }
    }
}

__global__ void euler_totient_diff_square(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_perfect_square(d_numbers[idx])) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

__global__ void euler_totient_diff_non_square(int* d_numbers, int size, int* d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && !is_perfect_square(d_numbers[idx])) {
        int num = d_numbers[idx];
        int totient = num;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                while (num % i == 0) num /= i;
                totient -= totient / i;
            }
        }
        if (num > 1) totient -= totient / num;
        d_results[idx] = totient - (num - 1);
    } else {
        d_results[idx] = 0;
    }
}

int main() {
    const int size = 1024;
    int *h_numbers, *d_numbers, *d_results;

    h_numbers = (int*)malloc(size * sizeof(int));
    cudaMalloc((void**)&d_numbers, size * sizeof(int));
    cudaMalloc((void**)&d_results, size * sizeof(int));

    generate_random_numbers(h_numbers, size);
    cudaMemcpy(d_numbers, h_numbers, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks(size / 256 + 1, 1);
    dim3 threads(256, 1);

    euler_totient_diff<<<blocks, threads>>>(d_numbers, size, d_results);

    cudaMemcpy(h_numbers, d_results, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        printf("Number: %d, Euler Totient Diff: %d\n", h_numbers[i], d_results[i]);
    }

    free(h_numbers);
    cudaFree(d_numbers);
    cudaFree(d_results);

    return 0;
}
