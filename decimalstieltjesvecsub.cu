#include <cuda_runtime.h>
#include <math.h>

__device__ bool is_prime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__global__ void find_primes(uint64_t *primes, uint64_t *count, uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start) {
        uint64_t num = start + idx;
        if (is_prime(num)) {
            atomicAdd(count, 1);
            int c = atomicAdd(count, 0);
            primes[c] = num;
        }
    }
}

__global__ void add_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = primes1[idx] + primes2[idx];
}

__global__ void multiply_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = primes1[idx] * primes2[idx];
}

__global__ void subtract_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = primes1[idx] - primes2[idx];
}

__global__ void mod_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = primes1[idx] % primes2[idx];
}

__global__ void power_primes(uint64_t *primes, uint64_t *exponents, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = pow(primes[idx], exponents[idx]);
}

__global__ void gcd_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = __gcd(primes1[idx], primes2[idx]);
}

__global__ void lcm_primes(uint64_t *primes1, uint64_t *primes2, uint64_t *result, uint64_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = (primes1[idx] / gcd_primes(primes1[idx], primes2[idx])) * primes2[idx];
}

__global__ void sieve_of_eratosthenes(uint32_t *is_prime, uint64_t limit) {
    extern __shared__ bool shared_is_prime[];
    for (uint32_t i = threadIdx.x; i < limit; i += blockDim.x)
        is_prime[i] = true;
    __syncthreads();
    for (uint32_t p = 2 + threadIdx.x; p * p <= limit; p += blockDim.x) {
        if (is_prime[p]) {
            for (uint32_t j = p * p; j < limit; j += p)
                is_prime[j] = false;
        }
    }
}

__global__ void count_primes(uint32_t *is_prime, uint64_t *count, uint64_t limit) {
    extern __shared__ bool shared_is_prime[];
    for (uint32_t i = threadIdx.x; i < limit; i += blockDim.x)
        if (is_prime[i])
            atomicAdd(count, 1);
}

__global__ void find_nth_prime(uint32_t *is_prime, uint64_t *nth_prime, uint64_t n, uint64_t limit) {
    extern __shared__ bool shared_is_prime[];
    for (uint32_t i = threadIdx.x; i < limit; i += blockDim.x)
        if (is_prime[i] && atomicAdd(nth_prime, 0) == n - 1)
            *nth_prime = i;
}

__global__ void generate_random_primes(uint64_t *primes, uint64_t *count, uint64_t size) {
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        if (is_prime(curand(&state) % (1 << 32)))
            atomicAdd(count, 1);
            int c = atomicAdd(count, 0);
            primes[c] = curand(&state) % (1 << 32);
}

__global__ void sort_primes(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        for (uint64_t j = 0; j < size - 1; j++)
            if (primes[j] > primes[j + 1]) {
                uint64_t temp = primes[j];
                primes[j] = primes[j + 1];
                primes[j + 1] = temp;
            }
}

__global__ void binary_search_primes(uint64_t *primes, uint64_t size, uint64_t target) {
    extern __shared__ bool shared_found[];
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (primes[mid] == target)
            shared_found[threadIdx.x] = true;
        else if (primes[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
}

__global__ void next_prime(uint64_t *primes, uint64_t size, uint64_t *next) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        if (i == size - 1 || primes[i] + 2 <= primes[i + 1])
            *next = primes[i] + 2;
}

__global__ void prev_prime(uint64_t *primes, uint64_t size, uint64_t *prev) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        if (i == 0 || primes[i] - 2 >= primes[i - 1])
            *prev = primes[i] - 2;
}

__global__ void twin_primes(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 1; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1])
            printf("%llu %llu\n", primes[i], primes[i + 1]);
}

__global__ void cousin_primes(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 1; i += blockDim.x * gridDim.x)
        if (abs(primes[i] - primes[i + 1]) == 4)
            printf("%llu %llu\n", primes[i], primes[i + 1]);
}

__global__ void sexy_primes(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 1; i += blockDim.x * gridDim.x)
        if (abs(primes[i] - primes[i + 1]) == 6)
            printf("%llu %llu\n", primes[i], primes[i + 1]);
}

__global__ void prime_gap(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 1; i += blockDim.x * gridDim.x)
        printf("%llu %llu\n", primes[i], primes[i + 1] - primes[i]);
}

__global__ void prime_triplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 2; i += blockDim.x * gridDim.x)
        if (primes[i] + 4 == primes[i + 1] && primes[i] + 6 == primes[i + 2])
            printf("%llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2]);
}

__global__ void prime_quadruplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 3; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 6 == primes[i + 2] && primes[i] + 8 == primes[i + 3])
            printf("%llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3]);
}

__global__ void prime_quintuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 4; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4])
            printf("%llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4]);
}

__global__ void prime_sextuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 5; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5])
            printf("%llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5]);
}

__global__ void prime_septuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 6; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6])
            printf("%llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6]);
}

__global__ void prime_octuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 7; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7]);
}

__global__ void prime_nonuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 8; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8]);
}

__global__ void prime_decuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 9; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9]);
}

__global__ void prime_undecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 10; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10]);
}

__global__ void prime_dodecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 11; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11]);
}

__global__ void prime_tridecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 12; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12]);
}

__global__ void prime_quattuordecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 13; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13]);
}

__global__ void prime_quindecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 14; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14]);
}

__global__ void prime_sexdecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 15; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14] && primes[i] + 30 == primes[i + 15])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14], primes[i + 15]);
}

__global__ void prime_septendecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 16; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14] && primes[i] + 30 == primes[i + 15] && primes[i] + 32 == primes[i + 16])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14], primes[i + 15], primes[i + 16]);
}

__global__ void prime_octodecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 17; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14] && primes[i] + 30 == primes[i + 15] && primes[i] + 32 == primes[i + 16] && primes[i] + 34 == primes[i + 17])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14], primes[i + 15], primes[i + 16], primes[i + 17]);
}

__global__ void prime_nonadecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 18; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14] && primes[i] + 30 == primes[i + 15] && primes[i] + 32 == primes[i + 16] && primes[i] + 34 == primes[i + 17] && primes[i] + 36 == primes[i + 18])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14], primes[i + 15], primes[i + 16], primes[i + 17], primes[i + 18]);
}

__global__ void prime_decadecuplet(uint64_t *primes, uint64_t size) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size - 19; i += blockDim.x * gridDim.x)
        if (primes[i] + 2 == primes[i + 1] && primes[i] + 4 == primes[i + 2] && primes[i] + 6 == primes[i + 3] && primes[i] + 8 == primes[i + 4] && primes[i] + 10 == primes[i + 5] && primes[i] + 12 == primes[i + 6] && primes[i] + 14 == primes[i + 7] && primes[i] + 16 == primes[i + 8] && primes[i] + 18 == primes[i + 9] && primes[i] + 20 == primes[i + 10] && primes[i] + 22 == primes[i + 11] && primes[i] + 24 == primes[i + 12] && primes[i] + 26 == primes[i + 13] && primes[i] + 28 == primes[i + 14] && primes[i] + 30 == primes[i + 15] && primes[i] + 32 == primes[i + 16] && primes[i] + 34 == primes[i + 17] && primes[i] + 36 == primes[i + 18] && primes[i] + 38 == primes[i + 19])
            printf("%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n", primes[i], primes[i + 1], primes[i + 2], primes[i + 3], primes[i + 4], primes[i + 5], primes[i + 6], primes[i + 7], primes[i + 8], primes[i + 9], primes[i + 10], primes[i + 11], primes[i + 12], primes[i + 13], primes[i + 14], primes[i + 15], primes[i + 16], primes[i + 17], primes[i + 18], primes[i + 19]);
}
