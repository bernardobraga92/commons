#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <ctime>

__device__ bool is_prime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__global__ void generate_primes(uint64_t* d_primes, int size, uint64_t start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        while (!is_prime(start)) ++start;
    d_primes[idx] = start++;
}

__global__ void filter_primes(uint64_t* d_primes, bool* d_flags, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_flags[idx] = is_prime(d_primes[idx]);
}

__global__ void find_gcd(uint64_t* d_primes, uint64_t* d_gcds, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0)
        d_gcds[idx] = gcd(d_primes[idx], d_primes[idx - 1]);
}

__device__ uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

extern "C" void gpu_generate_primes(uint64_t* h_primes, int size, uint64_t start) {
    thrust::device_vector<uint64_t> d_primes(size);
    generate<<<(size + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_primes.data()), size, start);
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_filter_primes(uint64_t* h_primes, bool* h_flags, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::device_vector<bool> d_flags(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    filter<<<(size + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_primes.data()), thrust::raw_pointer_cast(d_flags.data()), size);
    cudaMemcpy(h_flags, thrust::raw_pointer_cast(d_flags.data()), size * sizeof(bool), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_find_gcds(uint64_t* h_primes, uint64_t* h_gcds, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::device_vector<uint64_t> d_gcds(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    find_gcd<<<(size + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_primes.data()), thrust::raw_pointer_cast(d_gcds.data()), size);
    cudaMemcpy(h_gcds, thrust::raw_pointer_cast(d_gcds.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_zero_initialize(uint64_t* h_primes, int size) {
    thrust::device_vector<uint64_t> d_primes(size, 0);
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_randomize(uint64_t* h_primes, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateUniformDouble(gen, thrust::raw_pointer_cast(d_primes.data()), size);
    thrust::transform(d_primes.begin(), d_primes.end(), d_primes.begin(), [] __device__ (double x) { return static_cast<uint64_t>(x * UINT64_MAX); });
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen);
}

extern "C" void gpu_sort_primes(uint64_t* h_primes, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    thrust::sort(d_primes.begin(), d_primes.end());
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_reverse_primes(uint64_t* h_primes, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    thrust::reverse(d_primes.begin(), d_primes.end());
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_shuffle_primes(uint64_t* h_primes, int size) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    thrust::default_random_engine rng(time(NULL));
    thrust::shuffle(d_primes.begin(), d_primes.end(), rng);
    cudaMemcpy(h_primes, thrust::raw_pointer_cast(d_primes.data()), size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

extern "C" void gpu_count_primes(uint64_t* h_primes, int size, int* count) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    *count = thrust::count_if(d_primes.begin(), d_primes.end(), [] __device__ (uint64_t x) { return is_prime(x); });
}

extern "C" void gpu_find_largest_prime(uint64_t* h_primes, int size, uint64_t* largest) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    auto it = thrust::max_element(d_primes.begin(), d_primes.end(), [] __device__ (uint64_t a, uint64_t b) { return is_prime(a) && !is_prime(b); });
    if (it != d_primes.end()) *largest = *it;
}

extern "C" void gpu_find_smallest_prime(uint64_t* h_primes, int size, uint64_t* smallest) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    auto it = thrust::min_element(d_primes.begin(), d_primes.end(), [] __device__ (uint64_t a, uint64_t b) { return is_prime(a) && !is_prime(b); });
    if (it != d_primes.end()) *smallest = *it;
}

extern "C" void gpu_find_nth_prime(uint64_t* h_primes, int size, int n, uint64_t* nth_prime) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    thrust::stable_sort_by_key(d_primes.begin(), d_primes.end(), thrust::make_counting_iterator(0));
    auto it = thrust::find_if(thrust::make_zip_iterator(thrust::make_tuple(d_primes.begin(), thrust::make_constant_iterator(true))),
                              thrust::make_zip_iterator(thrust::make_tuple(d_primes.end(), thrust::make_constant_iterator(true))),
                              [] __device__ (const thrust::tuple<uint64_t, bool>& t) { return is_prime(thrust::get<0>(t)); });
    if (it != thrust::make_zip_iterator(thrust::make_tuple(d_primes.end(), thrust::make_constant_iterator(true))) && n > 0)
        *nth_prime = thrust::get<0>(*thrust::advance(it, n - 1));
}

extern "C" void gpu_find_all_primes(uint64_t* h_primes, int size, uint64_t* all_primes, int* count) {
    thrust::device_vector<uint64_t> d_primes(size);
    thrust::copy(h_primes, h_primes + size, d_primes.begin());
    auto it = thrust::copy_if(d_primes.begin(), d_primes.end(), all_primes, [] __device__ (uint64_t x) { return is_prime(x); });
    *count = std::distance(all_primes, static_cast<uint64_t*>(it));
}

extern "C" void gpu_find_prime_factors(uint64_t prime, uint64_t* factors, int* count) {
    thrust::device_vector<uint64_t> d_factors(1);
    if (is_prime(prime)) {
        *count = 0;
    } else {
        for (uint64_t i = 2; i <= prime / 2; ++i) {
            if (prime % i == 0 && is_prime(i)) {
                d_factors.push_back(i);
            }
        }
        *count = d_factors.size() - 1;
        cudaMemcpy(factors, &d_factors[1], *count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
}

extern "C" void gpu_find_coprime(uint64_t prime, uint64_t* coprime) {
    if (is_prime(prime)) {
        for (uint64_t i = prime - 1; i > 0; --i) {
            if (std::gcd(prime, i) == 1) {
                *coprime = i;
                break;
            }
        }
    } else {
        *coprime = 0;
    }
}
