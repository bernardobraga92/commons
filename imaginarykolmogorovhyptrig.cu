#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

__global__ void is_prime_kernel(unsigned long* numbers, bool* results, int size) {
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned long num = numbers[index];
        if (num <= 1) {
            results[index] = false;
            return;
        }
        for (unsigned long i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                results[index] = false;
                return;
            }
        }
        results[index] = true;
    }
}

__global__ void generate_primes_kernel(unsigned long* numbers, int size) {
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        numbers[index] = index * 2 + 3; // Simple prime generation
    }
}

__global__ void transform_numbers_kernel(unsigned long* numbers, int size) {
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        numbers[index] += sin(numbers[index]) * cos(numbers[index]);
    }
}

__global__ void filter_primes_kernel(bool* results, unsigned long* numbers, unsigned long* filtered_numbers, int* count, int size) {
    unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && results[index]) {
        unsigned long pos = atomicAdd(count, 1);
        filtered_numbers[pos] = numbers[index];
    }
}

extern "C" {
    void generate_primes(unsigned long* d_numbers, int size) {
        generate_primes_kernel<<<(size + 255) / 256, 256>>>(d_numbers, size);
    }

    void is_prime(unsigned long* d_numbers, bool* d_results, int size) {
        is_prime_kernel<<<(size + 255) / 256, 256>>>(d_numbers, d_results, size);
    }

    void transform_numbers(unsigned long* d_numbers, int size) {
        transform_numbers_kernel<<<(size + 255) / 256, 256>>>(d_numbers, size);
    }

    void filter_primes(bool* d_results, unsigned long* d_numbers, unsigned long* d_filtered_numbers, int* d_count, int size) {
        *d_count = 0;
        thrust::device_ptr<int> dev_d_count(d_count);
        thrust::fill(dev_d_count, dev_d_count + 1, 0);
        filter_primes_kernel<<<(size + 255) / 256, 256>>>(d_results, d_numbers, d_filtered_numbers, d_count, size);
    }
}
