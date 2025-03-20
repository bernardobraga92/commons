#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__global__ void findPrimes(unsigned long long start, unsigned long long end, bool* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end - start)
        results[idx] = isPrime(start + idx);
}

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void computeGCDs(int* data, int n, int* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = gcd(data[idx], data[(idx + 1) % n]);
}

__device__ double stieltjesFunction(unsigned int n) {
    if (n == 0) return 1.0;
    else return 1.0 / (n * n);
}

__global__ void computeStieltjesDiff(double* results, unsigned int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = stieltjesFunction(idx) - stieltjesFunction(idx + 1);
}

__device__ bool isDivisibleByStieltjes(unsigned long long num, double* stieltjesValues, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i)
        if (num % (int)(stieltjesValues[i] * 1e9) == 0)
            return true;
    return false;
}

__global__ void filterDivisibleByStieltjes(unsigned long long* data, unsigned int n, double* stieltjesValues, bool* results, unsigned int size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = isDivisibleByStieltjes(data[idx], stieltjesValues, size);
}

__device__ unsigned long long factorial(unsigned int n) {
    return (n == 0 || n == 1) ? 1 : n * factorial(n - 1);
}

__global__ void computeFactorials(unsigned int* data, unsigned int n, unsigned long long* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = factorial(data[idx]);
}

__device__ unsigned long long primeProduct(unsigned long long* primes, unsigned int size) {
    unsigned long long product = 1;
    for (unsigned int i = 0; i < size; ++i)
        product *= primes[i];
    return product;
}

__global__ void computePrimeProducts(unsigned long long* primes, unsigned int n, unsigned long long* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = primeProduct(&primes[idx], 1);
}

__device__ double stieltjesDifference(double a, double b) {
    return a - b;
}

__global__ void computeStieltjesDifferences(double* data, unsigned int n, double* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = stieltjesDifference(data[idx], data[(idx + 1) % n]);
}

__device__ unsigned long long sumOfPrimes(unsigned long long* primes, unsigned int size) {
    unsigned long long sum = 0;
    for (unsigned int i = 0; i < size; ++i)
        sum += primes[i];
    return sum;
}

__global__ void computeSumOfPrimes(unsigned long long* primes, unsigned int n, unsigned long long* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = sumOfPrimes(&primes[idx], 1);
}

__device__ bool isEven(unsigned long long num) {
    return num % 2 == 0;
}

__global__ void filterEvens(unsigned long long* data, unsigned int n, bool* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = isEven(data[idx]);
}

__device__ unsigned long long lcm(unsigned long long a, unsigned long long b) {
    return a / gcd(a, b) * b;
}

__global__ void computeLCMs(unsigned long long* data, unsigned int n, unsigned long long* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = lcm(data[idx], data[(idx + 1) % n]);
}

__device__ bool isDivisible(unsigned long long num, unsigned int divisor) {
    return num % divisor == 0;
}

__global__ void filterDivisibles(unsigned long long* data, unsigned int n, unsigned int divisor, bool* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = isDivisible(data[idx], divisor);
}

__device__ double stieltjesRatio(double a, double b) {
    return a / b;
}

__global__ void computeStieltjesRatios(double* data, unsigned int n, double* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = stieltjesRatio(data[idx], data[(idx + 1) % n]);
}

__device__ unsigned long long powerMod(unsigned long long base, unsigned int exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void computePowerMods(unsigned long long* data, unsigned int n, unsigned int exp, unsigned long long mod, unsigned long long* results) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        results[idx] = powerMod(data[idx], exp, mod);
}

int main() {
    const unsigned long long numPrimes = 1024;
    bool* d_results;
    unsigned long long* d_primes;
    cudaMalloc(&d_results, numPrimes * sizeof(bool));
    cudaMalloc(&d_primes, numPrimes * sizeof(unsigned long long));

    findPrimes<<<(numPrimes + 255) / 256, 256>>>(1000000000000ULL, 1000000000256ULL, d_results);
    cudaMemcpy(d_primes, d_results, numPrimes * sizeof(bool), cudaMemcpyDeviceToDevice);

    cudaFree(d_results);
    cudaFree(d_primes);
    return 0;
}
