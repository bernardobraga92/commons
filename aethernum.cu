#include <cuda_runtime.h>
#include <iostream>

__device__ int aether_is_prime(unsigned long long num) {
    if (num <= 1) return 0;
    for (unsigned long long i = 2; i * i <= num; ++i) {
        if (num % i == 0) return 0;
    }
    return 1;
}

__device__ unsigned long long aether_next_prime(unsigned long long start) {
    while (!aether_is_prime(start)) ++start;
    return start;
}

__device__ unsigned long long aether_random_prime(unsigned long long min, unsigned long long max) {
    unsigned long long num = (min + max) / 2;
    while (!aether_is_prime(num)) {
        if (num % 2 == 0) ++num;
        else --num;
    }
    return num;
}

__device__ unsigned long long aether_gcd(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        unsigned long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__device__ unsigned long long aether_lcm(unsigned long long a, unsigned long long b) {
    return a / aether_gcd(a, b) * b;
}

__global__ void aether_find_primes(unsigned long long* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) primes[idx] = aether_next_prime(idx + 2);
}

__device__ unsigned long long aether_power_mod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__device__ unsigned long long aether_factorial(unsigned long long num) {
    unsigned long long result = 1;
    for (unsigned long long i = 2; i <= num; ++i) {
        result *= i;
    }
    return result;
}

__device__ unsigned long long aether_fibonacci(int n) {
    if (n <= 1) return n;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

__global__ void aether_random_primes(unsigned long long* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) primes[idx] = aether_random_prime(idx * 1000000, (idx + 1) * 1000000);
}

__device__ unsigned long long aether_sum_of_primes(unsigned long long* primes, int count) {
    unsigned long long sum = 0;
    for (int i = 0; i < count; ++i) {
        if (aether_is_prime(primes[i])) sum += primes[i];
    }
    return sum;
}

__device__ unsigned long long aether_product_of_primes(unsigned long long* primes, int count) {
    unsigned long long product = 1;
    for (int i = 0; i < count; ++i) {
        if (aether_is_prime(primes[i])) product *= primes[i];
    }
    return product;
}

__global__ void aether_generate_primes(unsigned long long* primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) primes[idx] = aether_next_prime(idx * 1000000);
}

__device__ unsigned long long aether_largest_prime_factor(unsigned long long num) {
    unsigned long long largest = -1;
    while (num % 2 == 0) {
        largest = 2;
        num /= 2;
    }
    for (unsigned long long i = 3; i * i <= num; i += 2) {
        while (num % i == 0) {
            largest = i;
            num /= i;
        }
    }
    if (num > 2) largest = num;
    return largest;
}

__global__ void aether_find_largest_prime_factors(unsigned long long* nums, unsigned long long* factors, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) factors[idx] = aether_largest_prime_factor(nums[idx]);
}

__device__ unsigned long long aether_modular_inverse(unsigned long long a, unsigned long long m) {
    unsigned long long m0 = m, t, q;
    unsigned long long x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m, a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

__global__ void aether_compute_modular_inverses(unsigned long long* nums, unsigned long long* inverses, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) inverses[idx] = aether_modular_inverse(nums[idx], 1000000007);
}

__device__ unsigned long long aether_phi(unsigned long long n) {
    unsigned long long result = n;
    for (unsigned long long p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            while (n % p == 0) n /= p;
            result -= result / p;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}

__global__ void aether_compute_totients(unsigned long long* nums, unsigned long long* totients, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) totients[idx] = aether_phi(nums[idx]);
}

__device__ unsigned long long aether_legendre_symbol(int a, int p) {
    int ls = 1;
    a %= p;
    while (a != 0) {
        if (a < 0) {
            a = -a;
            ls *= (-1) % p;
        }
        while (a % 2 == 0) {
            a /= 2;
            if ((p % 8) == 3 || (p % 8) == 7) ls *= (-1) % p;
        }
        if (a == 1) break;
        std::swap(a, p);
        if ((a % 4) == 3 && (p % 4) == 3) ls *= (-1) % p;
    }
    return ls;
}

__global__ void aether_compute_legendre_symbols(int* nums, int p, int* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) results[idx] = aether_legendre_symbol(nums[idx], p);
}
