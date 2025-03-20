#include <cuda_runtime.h>
#include <stdio.h>

__device__ int is_prime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void find_primes(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit && is_prime(idx)) {
        atomicAdd(found, 1);
        primes[atomicAdd(found, -1)] = idx;
    }
}

__global__ void markov_step(int *state, int *next_state, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        next_state[idx] = (state[idx] * 23 + 457) % 1009;
    }
}

__global__ void permute(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int swap_idx = idx + ((blockIdx.y * blockDim.y + threadIdx.y) % (size - idx));
        int temp = data[idx];
        data[idx] = data[swap_idx];
        data[swap_idx] = temp;
    }
}

__global__ void generate_random_primes(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 2 + idx * 31 + threadIdx.y * 17;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}

__global__ void filter_primes(int *input, int *output, int *count, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(input[idx])) {
        atomicAdd(count, 1);
        output[atomicAdd(count, -1)] = input[idx];
    }
}

__global__ void transform_primes(int *primes, int *transformed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        transformed[idx] = primes[idx] * 37 + 123;
    }
}

__global__ void reduce_primes(int *primes, int *sum, int size) {
    extern __shared__ int shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        shared[threadIdx.x] = primes[idx];
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(sum, shared[0]);
}

__global__ void sort_primes(int *primes, int size) {
    for (int stride = size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size - stride && primes[idx] > primes[idx + stride]) {
            int temp = primes[idx];
            primes[idx] = primes[idx + stride];
            primes[idx + stride] = temp;
        }
    }
}

__global__ void shuffle_primes(int *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int swap_idx = idx + ((blockIdx.y * blockDim.y + threadIdx.y) % (size - idx));
        int temp = primes[idx];
        primes[idx] = primes[swap_idx];
        primes[swap_idx] = temp;
    }
}

__global__ void merge_primes(int *primes1, int *primes2, int *merged, int size1, int size2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size1) {
        merged[idx] = primes1[idx];
    }
    if (idx < size2) {
        merged[size1 + idx] = primes2[idx];
    }
}

__global__ void sieve_primes(int *primes, int *is_prime_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        is_prime_array[idx] = 1;
    }
    __syncthreads();

    for (int i = idx; i < size; i += size) {
        if (is_prime_array[i]) {
            for (int j = i * i; j < size; j += i) {
                is_prime_array[j] = 0;
            }
        }
    }
}

__global__ void find_largest_prime(int *primes, int *largest_prime, int size) {
    extern __shared__ int shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        shared[threadIdx.x] = primes[idx];
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && shared[threadIdx.x] < shared[threadIdx.x + s]) {
            shared[threadIdx.x] = shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicMax(largest_prime, shared[0]);
}

__global__ void generate_fibonacci_primes(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int a = 1, b = 1, fib = 1;
        for (int i = 0; i <= idx; i++) {
            fib = a + b;
            a = b;
            b = fib;
        }
        if (is_prime(fib)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = fib;
        }
    }
}

__global__ void count_primes(int *primes, int *count, int size) {
    extern __shared__ int shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        shared[threadIdx.x] = is_prime(primes[idx]);
    } else {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(count, shared[0]);
}

__global__ void generate_random_primes_fast(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 2 + idx * 31 + threadIdx.y * 17;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}

__global__ void generate_random_primes_fast2(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 3 + idx * 47 + threadIdx.y * 29;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}

__global__ void generate_random_primes_fast3(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 5 + idx * 61 + threadIdx.y * 37;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}

__global__ void generate_random_primes_fast4(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 7 + idx * 73 + threadIdx.y * 41;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}

__global__ void generate_random_primes_fast5(int *primes, int *found, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        int candidate = 11 + idx * 89 + threadIdx.y * 47;
        if (is_prime(candidate)) {
            atomicAdd(found, 1);
            primes[atomicAdd(found, -1)] = candidate;
        }
    }
}
