#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <iostream>

__device__ bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i)
        if (num % i == 0) return false;
    return true;
}

__global__ void find_primes(int* d_numbers, bool* d_results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_results[idx] = is_prime(d_numbers[idx]);
}

void generate_large_primes(thrust::host_vector<int>& h_numbers, thrust::device_vector<bool>& d_results, int count) {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(1000000, 2000000);
    
    for (int i = 0; i < count; ++i)
        h_numbers[i] = dist(rng);

    thrust::copy(h_numbers.begin(), h_numbers.end(), d_results.begin());

    find_primes<<<(count + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_results.data()), thrust::raw_pointer_cast(d_results.data() + count), count);
}

int main() {
    int count = 100;
    thrust::host_vector<int> h_numbers(count);
    thrust::device_vector<bool> d_results(count);

    generate_large_primes(h_numbers, d_results, count);

    for (int i = 0; i < count; ++i)
        if (h_numbers[i] == 1 || h_numbers[i] == 2)
            std::cout << "Prime: " << h_numbers[i] << std::endl;

    return 0;
}
