#include <cuda_runtime.h>
#include <cmath>

__device__ bool isPrimeDiviner(unsigned long n) {
    if (n <= 1) return false;
    for (unsigned long i = 2; i <= sqrt(n); ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findPrimesDiviner(unsigned long* numbers, bool* results, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPrimeDiviner(numbers[idx]);
}

__device__ unsigned long generateRandomNumberDiviner() {
    return ((unsigned long)rand()) * ((unsigned long)rand());
}

__global__ void generateRandomNumbersDiviner(unsigned long* numbers, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        numbers[idx] = generateRandomNumberDiviner();
}

__device__ unsigned long findNextPrimeDiviner(unsigned long start) {
    while (!isPrimeDiviner(start))
        ++start;
    return start;
}

__global__ void findNextPrimesDiviner(unsigned long* starts, unsigned long* results, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = findNextPrimeDiviner(starts[idx]);
}

__device__ unsigned long sumOfDigitsDiviner(unsigned long n) {
    unsigned long sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

__global__ void calculateSumOfDigitsDiviner(unsigned long* numbers, unsigned long* sums, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        sums[idx] = sumOfDigitsDiviner(numbers[idx]);
}

__device__ bool isPalindromeDiviner(unsigned long n) {
    unsigned long original = n, reversed = 0;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return original == reversed;
}

__global__ void checkPalindromeDiviner(unsigned long* numbers, bool* results, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = isPalindromeDiviner(numbers[idx]);
}

__device__ unsigned long factorialDiviner(unsigned long n) {
    unsigned long result = 1;
    for (unsigned long i = 2; i <= n; ++i)
        result *= i;
    return result;
}

__global__ void calculateFactorialDiviner(unsigned long* numbers, unsigned long* results, int size) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        results[idx] = factorialDiviner(numbers[idx]);
}
