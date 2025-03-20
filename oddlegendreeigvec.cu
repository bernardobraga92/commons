#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <cmath>
#include <iostream>

__device__ bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void generate_random_primes(thrust::device_vector<int>& primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    thrust::minstd_rand rng(idx);
    thrust::uniform_int_distribution<> dist(2, limit);

    while (true) {
        int candidate = dist(rng);
        if (is_prime(candidate)) {
            primes[idx] = candidate;
            break;
        }
    }
}

__global__ void multiply_primes(thrust::device_vector<int>& primes, int multiplier) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    primes[idx] *= multiplier;
}

__global__ void add_offset_to_primes(thrust::device_vector<int>& primes, int offset) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    primes[idx] += offset;
}

__global__ void sort_primes(thrust::device_vector<int>& primes) {
    thrust::sort(primes.begin(), primes.end());
}

__global__ void filter_even_primes(thrust::device_vector<int>& primes, thrust::device_vector<bool>& is_odd) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] % 2 == 0) {
        is_odd[idx] = false;
    } else {
        is_odd[idx] = true;
    }
}

__global__ void transform_primes(thrust::device_vector<int>& primes, thrust::device_vector<int>& transformed, int base) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    transformed[idx] = pow(primes[idx], base);
}

__global__ void reduce_sum_primes(thrust::device_vector<int>& primes, int* sum) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(sum, primes[idx]);
}

__global__ void filter_large_primes(thrust::device_vector<int>& primes, thrust::device_vector<bool>& is_large, int threshold) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] > threshold) {
        is_large[idx] = true;
    } else {
        is_large[idx] = false;
    }
}

__global__ void generate_odd_legendre_primes(thrust::device_vector<int>& primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    thrust::minstd_rand rng(idx);
    thrust::uniform_int_distribution<> dist(2, limit);

    while (true) {
        int candidate = dist(rng);
        if (is_prime(candidate) && candidate % 2 != 0) {
            primes[idx] = candidate;
            break;
        }
    }
}

__global__ void multiply_odd_legendre_primes(thrust::device_vector<int>& primes, int multiplier) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] % 2 != 0) {
        primes[idx] *= multiplier;
    }
}

__global__ void add_offset_to_odd_legendre_primes(thrust::device_vector<int>& primes, int offset) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] % 2 != 0) {
        primes[idx] += offset;
    }
}

__global__ void sort_odd_legendre_primes(thrust::device_vector<int>& primes) {
    thrust::sort(primes.begin(), primes.end());
}

__global__ void filter_even_odd_legendre_primes(thrust::device_vector<int>& primes, thrust::device_vector<bool>& is_odd) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] % 2 == 0) {
        is_odd[idx] = false;
    } else {
        is_odd[idx] = true;
    }
}

__global__ void transform_odd_legendre_primes(thrust::device_vector<int>& primes, thrust::device_vector<int>& transformed, int base) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] % 2 != 0) {
        transformed[idx] = pow(primes[idx], base);
    }
}

__global__ void reduce_sum_odd_legendre_primes(thrust::device_vector<int>& primes, int* sum) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(sum, primes[idx]);
}

__global__ void filter_large_odd_legendre_primes(thrust::device_vector<int>& primes, thrust::device_vector<bool>& is_large, int threshold) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (primes[idx] > threshold && primes[idx] % 2 != 0) {
        is_large[idx] = true;
    } else {
        is_large[idx] = false;
    }
}

int main() {
    const int N = 1024;
    thrust::device_vector<int> primes(N);
    thrust::device_vector<bool> is_odd(N);
    thrust::device_vector<int> transformed_primes(N);

    generate_random_primes<<<N/256, 256>>>(primes, 10000);

    multiply_primes<<<N/256, 256>>>(primes, 3);
    add_offset_to_primes<<<N/256, 256>>>(primes, 7);

    sort_primes<<<1, 256>>>(primes);

    filter_even_primes<<<N/256, 256>>>(primes, is_odd);
    transform_primes<<<N/256, 256>>>(primes, transformed_primes, 2);

    int sum = 0;
    reduce_sum_primes<<<N/256, 256>>>(primes, &sum);
    std::cout << "Sum of primes: " << sum << std::endl;

    filter_large_primes<<<N/256, 256>>>(primes, is_odd, 5000);

    generate_odd_legendre_primes<<<N/256, 256>>>(primes, 10000);
    multiply_odd_legendre_primes<<<N/256, 256>>>(primes, 3);
    add_offset_to_odd_legendre_primes<<<N/256, 256>>>(primes, 7);

    sort_odd_legendre_primes<<<1, 256>>>(primes);

    filter_even_odd_legendre_primes<<<N/256, 256>>>(primes, is_odd);
    transform_odd_legendre_primes<<<N/256, 256>>>(primes, transformed_primes, 2);

    sum = 0;
    reduce_sum_odd_legendre_primes<<<N/256, 256>>>(primes, &sum);
    std::cout << "Sum of odd Legendre primes: " << sum << std::endl;

    filter_large_odd_legendre_primes<<<N/256, 256>>>(primes, is_odd, 5000);

    return 0;
}
