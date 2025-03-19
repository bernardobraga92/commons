#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

__device__ bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void find_primes_in_range(int* d_start, int* d_end, bool* d_is_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_prime[idx] = is_prime(d_start[idx]);
}

__device__ bool is_tarski_prime(int n) {
    for (int i = 2; i <= sqrt(n); ++i)
        if (is_prime(i) && n % i == 0)
            return false;
    return true;
}

__global__ void find_tarski_primes_in_range(int* d_start, int* d_end, bool* d_is_tarski_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_tarski_prime[idx] = is_tarski_prime(d_start[idx]);
}

__device__ int next_prime(int n) {
    while (!is_prime(++n));
    return n;
}

__global__ void find_next_primes(int* d_numbers, int* d_next_primes, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_next_primes[idx] = next_prime(d_numbers[idx]);
}

__device__ bool is_coprime(int a, int b) {
    for (int i = 2; i <= min(a, b); ++i)
        if (a % i == 0 && b % i == 0)
            return false;
    return true;
}

__global__ void check_coprimes(int* d_numbers1, int* d_numbers2, bool* d_is_coprime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_coprime[idx] = is_coprime(d_numbers1[idx], d_numbers2[idx]);
}

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void compute_gcds(int* d_numbers1, int* d_numbers2, int* d_gcds, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_gcds[idx] = gcd(d_numbers1[idx], d_numbers2[idx]);
}

__device__ bool is_safe_prime(int n) {
    return is_prime(n) && is_prime((n - 1) / 2);
}

__global__ void find_safe_primes_in_range(int* d_start, int* d_end, bool* d_is_safe_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_safe_prime[idx] = is_safe_prime(d_start[idx]);
}

__device__ int largest_prime_factor(int n) {
    int largest = -1;
    while (n % 2 == 0) {
        largest = 2;
        n >>= 1;
    }
    for (int i = 3; i <= sqrt(n); i += 2)
        while (n % i == 0) {
            largest = i;
            n /= i;
        }
    if (n > 2)
        largest = n;
    return largest;
}

__global__ void find_largest_prime_factors(int* d_numbers, int* d_largest_prime_factors, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_largest_prime_factors[idx] = largest_prime_factor(d_numbers[idx]);
}

__device__ bool is_carmichael_number(int n) {
    for (int i = 2; i < n; ++i)
        if (gcd(i, n) == 1 && modpow(i, n - 1, n) != 1)
            return false;
    return true;
}

__global__ void find_carmichael_numbers_in_range(int* d_start, int* d_end, bool* d_is_carmichael_number, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_carmichael_number[idx] = is_carmichael_number(d_start[idx]);
}

__device__ int euler_totient(int n) {
    int result = n;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    if (n > 1)
        result -= result / n;
    return result;
}

__global__ void compute_euler_totients(int* d_numbers, int* d_euler_totients, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_euler_totients[idx] = euler_totient(d_numbers[idx]);
}

__device__ bool is_square_free(int n) {
    for (int i = 2; i * i <= n; ++i)
        if (n % (i * i) == 0)
            return false;
    return true;
}

__global__ void find_square_free_in_range(int* d_start, int* d_end, bool* d_is_square_free, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_square_free[idx] = is_square_free(d_start[idx]);
}

__device__ int primitive_root_modulo(int n) {
    for (int r = 2; r <= n; ++r) {
        bool flag = true;
        for (int i = 1; i < n && flag; ++i)
            if (modpow(r, i, n) == 1 && i != n - 1)
                flag = false;
        if (flag) return r;
    }
    return -1;
}

__global__ void find_primitive_roots_modulo(int* d_numbers, int* d_primitive_roots, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_primitive_roots[idx] = primitive_root_modulo(d_numbers[idx]);
}

__device__ bool is_super_prime(int n) {
    return is_prime(n) && is_prime(prime_count_below(n));
}

__global__ void find_super_primes_in_range(int* d_start, int* d_end, bool* d_is_super_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_super_prime[idx] = is_super_prime(d_start[idx]);
}

__device__ int prime_count_below(int n) {
    int count = 0;
    for (int i = 2; i <= n; ++i)
        if (is_prime(i))
            ++count;
    return count;
}

__global__ void compute_prime_counts(int* d_numbers, int* d_prime_counts, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_prime_counts[idx] = prime_count_below(d_numbers[idx]);
}

__device__ bool is_fibonacci_prime(int n) {
    for (int i = 2; i <= n; ++i) {
        int a = 0, b = 1;
        while (b < n) {
            int temp = b;
            b += a;
            a = temp;
        }
        if (b == n && is_prime(n))
            return true;
    }
    return false;
}

__global__ void find_fibonacci_primes_in_range(int* d_start, int* d_end, bool* d_is_fibonacci_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_fibonacci_prime[idx] = is_fibonacci_prime(d_start[idx]);
}

__device__ bool is_twin_prime(int n) {
    return is_prime(n) && (is_prime(n - 2) || is_prime(n + 2));
}

__global__ void find_twin_primes_in_range(int* d_start, int* d_end, bool* d_is_twin_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_is_twin_prime[idx] = is_twin_prime(d_start[idx]);
}

__device__ int modpow(int base, int exp, int mod) {
    base = base % mod;
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void find_modpow_results(int* d_bases, int* d_exps, int* d_mods, int* d_results, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        d_results[idx] = modpow(d_bases[idx], d_exps[idx], d_mods[idx]);
}
