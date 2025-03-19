#include <cuda_runtime.h>
#include <iostream>

__global__ void euclid_prime_check(int *is_prime, int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 1 && i <= num / 2) {
        if (num % i == 0) is_prime[0] = 0;
    }
}

__global__ void euclid_prime_generate(int *primes, int limit, int index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < limit && primes[i] == 1) {
        int num = 2 * i + 3;
        for (int j = 0; j < index; j++) {
            if (num % primes[j] == 0) {
                primes[i] = 0;
                break;
            }
        }
    }
}

__global__ void euclid_prime_sieve(int *primes, int limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < limit && i % 2 != 0) primes[i] = 1;
}

__global__ void euclid_prime_next(int *prime, int *is_prime, int limit) {
    for (int i = *prime + 2; i <= limit; i += 2) {
        __syncthreads();
        if (__single_thread_first_lane(is_prime[0])) break;
        *prime = i;
    }
}

__global__ void euclid_prime_sum(int *primes, int limit, int *sum) {
    extern __shared__ int shared_sum[];
    int tid = threadIdx.x;
    shared_sum[tid] = 0;
    for (int i = tid; i < limit; i += blockDim.x) {
        if (primes[i]) shared_sum[tid] += 2 * i + 3;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) *sum = shared_sum[0] - 3;
}

__global__ void euclid_prime_product(int *primes, int limit, int *product) {
    extern __shared__ int shared_product[];
    int tid = threadIdx.x;
    shared_product[tid] = 1;
    for (int i = tid; i < limit; i += blockDim.x) {
        if (primes[i]) shared_product[tid] *= (2 * i + 3);
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_product[tid] *= shared_product[tid + s];
        __syncthreads();
    }
    if (tid == 0) *product = shared_product[0];
}

__global__ void euclid_prime_count(int *primes, int limit, int *count) {
    extern __shared__ int shared_count[];
    int tid = threadIdx.x;
    shared_count[tid] = 0;
    for (int i = tid; i < limit; i += blockDim.x) {
        if (primes[i]) shared_count[tid]++;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_count[tid] += shared_count[tid + s];
        __syncthreads();
    }
    if (tid == 0) *count = shared_count[0];
}

__global__ void euclid_prime_largest(int *primes, int limit, int *largest) {
    extern __shared__ int shared_largest[];
    int tid = threadIdx.x;
    shared_largest[tid] = 1;
    for (int i = tid; i < limit; i += blockDim.x) {
        if (primes[i]) shared_largest[tid] = 2 * i + 3;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_largest[tid] < shared_largest[tid + s]) shared_largest[tid] = shared_largest[tid + s];
        __syncthreads();
    }
    if (tid == 0) *largest = shared_largest[0];
}

__global__ void euclid_prime_smallest(int *primes, int limit, int *smallest) {
    extern __shared__ int shared_smallest[];
    int tid = threadIdx.x;
    shared_smallest[tid] = INT_MAX;
    for (int i = tid; i < limit; i += blockDim.x) {
        if (primes[i]) shared_smallest[tid] = 2 * i + 3;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_smallest[tid] > shared_smallest[tid + s]) shared_smallest[tid] = shared_smallest[tid + s];
        __syncthreads();
    }
    if (tid == 0) *smallest = shared_smallest[0];
}

__global__ void euclid_prime_random(int *primes, int limit, unsigned int seed, int *random_prime) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < limit && primes[tid]) {
        curandState state;
        curand_init(seed, tid, 0, &state);
        int random_num = curand(&state) % (limit - tid) + tid;
        if (primes[random_num]) *random_prime = 2 * random_num + 3;
    }
}

__global__ void euclid_prime_isqrt(int num, int *result) {
    int low = 0, high = num;
    while (low <= high) {
        int mid = (low + high) / 2;
        if ((mid + 1) * (mid + 1) > num) {
            *result = mid;
            break;
        }
        low = mid + 1;
    }
}

__global__ void euclid_prime_twin(int *primes, int limit, int *twin_count) {
    extern __shared__ int shared_twin_count[];
    int tid = threadIdx.x;
    shared_twin_count[tid] = 0;
    for (int i = tid; i < limit - 1; i += blockDim.x) {
        if (primes[i] && primes[i + 1]) shared_twin_count[tid]++;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_twin_count[tid] += shared_twin_count[tid + s];
        __syncthreads();
    }
    if (tid == 0) *twin_count = shared_twin_count[0];
}

__global__ void euclid_prime_safe(int *primes, int limit, int *safe_count) {
    extern __shared__ int shared_safe_count[];
    int tid = threadIdx.x;
    shared_safe_count[tid] = 0;
    for (int i = tid; i < limit - 1; i += blockDim.x) {
        if (primes[i] && primes[2 * i + 2]) shared_safe_count[tid]++;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_safe_count[tid] += shared_safe_count[tid + s];
        __syncthreads();
    }
    if (tid == 0) *safe_count = shared_safe_count[0];
}

__global__ void euclid_prime_cousin(int *primes, int limit, int *cousin_count) {
    extern __shared__ int shared_cousin_count[];
    int tid = threadIdx.x;
    shared_cousin_count[tid] = 0;
    for (int i = tid; i < limit - 2; i += blockDim.x) {
        if (primes[i] && primes[2 * i + 4]) shared_cousin_count[tid]++;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_cousin_count[tid] += shared_cousin_count[tid + s];
        __syncthreads();
    }
    if (tid == 0) *cousin_count = shared_cousin_count[0];
}

__global__ void euclid_prime_alexandrian(int *primes, int limit, int *alexandrian_count) {
    extern __shared__ int shared_alexandrian_count[];
    int tid = threadIdx.x;
    shared_alexandrian_count[tid] = 0;
    for (int i = tid; i < limit - 3; i += blockDim.x) {
        if (primes[i] && primes[2 * i + 6]) shared_alexandrian_count[tid]++;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_alexandrian_count[tid] += shared_alexandrian_count[tid + s];
        __syncthreads();
    }
    if (tid == 0) *alexandrian_count = shared_alexandrian_count[0];
}

int main() {
    return 0;
}
