#include <cuda_runtime.h>
#include <math.h>

__device__ bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void find_primes_in_range(int* primes, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = start + idx;
    if (is_prime(num)) {
        atomicAdd(primes, num);
    }
}

__global__ void generate_random_numbers(unsigned int* random_numbers, unsigned int seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, 0, &state);
    random_numbers[tid] = curand(&state);
}

__device__ bool is_catalan_prime(int num) {
    return is_prime(num) && (num == ((1 << (2 * num)) + 1));
}

__global__ void find_catalan_primes(unsigned int* catalan_primes, unsigned int* random_numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (is_catalan_prime(random_numbers[idx])) {
            atomicAdd(catalan_primes, random_numbers[idx]);
        }
    }
}

__global__ void filter_even_primes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] % 2 == 0) {
        atomicAdd(primes, 1);
    }
}

__global__ void filter_odd_primes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && primes[idx] % 2 != 0) {
        atomicAdd(primes, 1);
    }
}

__global__ void find_twin_primes(int* twin_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1 && is_prime(primes[idx] + 2)) {
        atomicAdd(twin_primes, primes[idx]);
    }
}

__global__ void find_safe_primes(int* safe_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime((primes[idx] - 1) / 2)) {
        atomicAdd(safe_primes, primes[idx]);
    }
}

__global__ void find_emirp_primes(int* emirp_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(reverse_number(primes[idx])) && reverse_number(primes[idx]) != primes[idx]) {
        atomicAdd(emirp_primes, primes[idx]);
    }
}

__global__ void find_pierpont_primes(int* pierpont_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime((3 << primes[idx]) - 1)) {
        atomicAdd(pierpont_primes, primes[idx]);
    }
}

__global__ void find_wilson_primes(int* wilson_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(factorial(primes[idx] - 1) % primes[idx] == primes[idx] - 1)) {
        atomicAdd(wilson_primes, primes[idx]);
    }
}

__global__ void find_fermat_primes(int* fermat_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime((1 << (1 << primes[idx])) + 1)) {
        atomicAdd(fermat_primes, primes[idx]);
    }
}

__global__ void find_mersenne_primes(int* mersenne_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime((1 << primes[idx]) - 1)) {
        atomicAdd(mersenne_primes, primes[idx]);
    }
}

__global__ void find_sophie_germain_primes(int* sophie_germain_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(2 * primes[idx] + 1)) {
        atomicAdd(sophie_germain_primes, primes[idx]);
    }
}

__global__ void find_cunningham_chains(int* cunningham_chains, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1 && is_prime(2 * primes[idx] + 1)) {
        atomicAdd(cunningham_chains, primes[idx]);
    }
}

__global__ void find_balanced_primes(int* balanced_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size - 1 && is_prime((primes[idx] + primes[idx - 1]) / 2)) {
        atomicAdd(balanced_primes, primes[idx]);
    }
}

__global__ void find_left_truncatable_primes(int* left_truncatable_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(truncate_number(primes[idx], 'L')) {
        atomicAdd(left_truncatable_primes, primes[idx]);
    }
}

__global__ void find_right_truncatable_primes(int* right_truncatable_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(truncate_number(primes[idx], 'R')) {
        atomicAdd(right_truncatable_primes, primes[idx]);
    }
}

__global__ void find_elliptic_curve_primes(int* elliptic_curve_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(4 * primes[idx] + 1)) {
        atomicAdd(elliptic_curve_primes, primes[idx]);
    }
}

__global__ void find_fibonacci_primes(int* fibonacci_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(fibonacci_number(primes[idx]))) {
        atomicAdd(fibonacci_primes, primes[idx]);
    }
}

__global__ void find_lychrel_numbers(int* lychrel_numbers, int* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_lychrel_number(numbers[idx])) {
        atomicAdd(lychrel_numbers, numbers[idx]);
    }
}

__global__ void find_pentagonal_primes(int* pentagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(pentagonal_number(primes[idx]))) {
        atomicAdd(pentagonal_primes, primes[idx]);
    }
}

__global__ void find_hexagonal_primes(int* hexagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(hexagonal_number(primes[idx]))) {
        atomicAdd(hexagonal_primes, primes[idx]);
    }
}

__global__ void find_heptagonal_primes(int* heptagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(heptagonal_number(primes[idx]))) {
        atomicAdd(heptagonal_primes, primes[idx]);
    }
}

__global__ void find_octagonal_primes(int* octagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(octagonal_number(primes[idx]))) {
        atomicAdd(octagonal_primes, primes[idx]);
    }
}

__global__ void find_nonagonal_primes(int* nonagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(nonagonal_number(primes[idx]))) {
        atomicAdd(nonagonal_primes, primes[idx]);
    }
}

__global__ void find_decagonal_primes(int* decagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(decagonal_number(primes[idx]))) {
        atomicAdd(decagonal_primes, primes[idx]);
    }
}

__global__ void find_dodecagonal_primes(int* dodecagonal_primes, int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && is_prime(dodecagonal_number(primes[idx]))) {
        atomicAdd(dodecagonal_primes, primes[idx]);
    }
}
