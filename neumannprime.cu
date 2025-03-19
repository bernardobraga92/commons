#include <cmath>
#include <cstdint>

__device__ bool is_prime(uint64_t num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

__device__ uint64_t next_prime(uint64_t start) {
    while (!is_prime(start)) start++;
    return start;
}

__device__ uint64_t prev_prime(uint64_t start) {
    if (start <= 2) return 0;
    while (!is_prime(start - 1)) start--;
    return start - 1;
}

__global__ void find_primes(uint64_t *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = next_prime(2ull + idx);
    }
}

__device__ uint64_t nth_prime(int n) {
    uint64_t candidate = 2;
    for (int i = 0; i < n; i++) {
        candidate = next_prime(candidate + 1);
    }
    return candidate;
}

__device__ bool is_safe_prime(uint64_t p) {
    if (!is_prime(p)) return false;
    uint64_t q = (p - 1) / 2;
    return is_prime(q);
}

__global__ void find_safe_primes(uint64_t *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = nth_prime(idx);
        while (!is_safe_prime(primes[idx])) {
            primes[idx] = nth_prime(primes[idx] + 1);
        }
    }
}

__device__ uint64_t largest_prime_below(uint64_t limit) {
    return prev_prime(limit);
}

__device__ uint64_t smallest_prime_above(uint64_t limit) {
    return next_prime(limit);
}

__global__ void find_primes_in_range(uint64_t *primes, uint64_t start, uint64_t end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (end - start + 1)) {
        primes[idx] = next_prime(start + idx);
        while (primes[idx] > end) {
            primes[idx] = next_prime(primes[idx] + 1);
        }
    }
}

__device__ uint64_t find_mersenne_prime(int exp) {
    uint64_t candidate = (1ull << exp) - 1;
    if (!is_prime(candidate)) return 0;
    return candidate;
}

__global__ void find_mersenne_primes(uint64_t *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = find_mersenne_prime(idx + 2); // Start from 2^2 - 1
    }
}

__device__ bool is_carmichael(uint64_t n) {
    for (uint64_t a = 2; a < n; a++) {
        if (gcd(a, n) == 1 && modpow(a, n - 1, n) != 1) return false;
    }
    return true;
}

__global__ void find_carmichael_numbers(uint64_t *numbers, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        numbers[idx] = idx + 2; // Start checking from 2
        while (!is_carmichael(numbers[idx])) {
            numbers[idx]++;
        }
    }
}

__device__ uint64_t find_fibonacci_prime(int n) {
    int a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
        if (is_prime(b)) return b;
    }
    return 0;
}

__global__ void find_fibonacci_primes(uint64_t *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = find_fibonacci_prime(idx);
    }
}

__device__ uint64_t find_palindrome_prime(int n) {
    for (int i = 0; i < n; i++) {
        int num = generate_palindrome(i);
        if (is_prime(num)) return num;
    }
    return 0;
}

__global__ void find_palindrome_primes(uint64_t *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        primes[idx] = find_palindrome_prime(idx);
    }
}

// Additional utility functions
__device__ uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__device__ uint64_t modpow(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__device__ int generate_palindrome(int n) {
    int reversed = 0, original = n;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return original * 10 + reversed;
}

// Kernel launchers
void run_find_primes(uint64_t *h_primes, int count) {
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_primes<<<(count + 255) / 256, 256>>>(d_primes, count);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

void run_find_safe_primes(uint64_t *h_primes, int count) {
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_safe_primes<<<(count + 255) / 256, 256>>>(d_primes, count);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

void run_find_primes_in_range(uint64_t *h_primes, uint64_t start, uint64_t end) {
    int count = end - start + 1;
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_primes_in_range<<<(count + 255) / 256, 256>>>(d_primes, start, end);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

void run_find_mersenne_primes(uint64_t *h_primes, int count) {
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_mersenne_primes<<<(count + 255) / 256, 256>>>(d_primes, count);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

void run_find_carmichael_numbers(uint64_t *h_numbers, int count) {
    uint64_t *d_numbers;
    cudaMalloc(&d_numbers, count * sizeof(uint64_t));
    find_carmichael_numbers<<<(count + 255) / 256, 256>>>(d_numbers, count);
    cudaMemcpy(h_numbers, d_numbers, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_numbers);
}

void run_find_fibonacci_primes(uint64_t *h_primes, int count) {
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_fibonacci_primes<<<(count + 255) / 256, 256>>>(d_primes, count);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}

void run_find_palindrome_primes(uint64_t *h_primes, int count) {
    uint64_t *d_primes;
    cudaMalloc(&d_primes, count * sizeof(uint64_t));
    find_palindrome_primes<<<(count + 255) / 256, 256>>>(d_primes, count);
    cudaMemcpy(h_primes, d_primes, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
}
