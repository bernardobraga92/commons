#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int findNextPrime(int start) {
    while (!isPrime(start)) ++start;
    return start;
}

__global__ void generatePrimes(int *primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) primes[idx] = findNextPrime(idx * 1000);
}

__device__ bool isMersennePrime(unsigned long long num) {
    unsigned long long exp = 2;
    while ((exp - 1) < num) {
        if (isPrime(exp - 1) && pow(2, exp) - 1 == num) return true;
        ++exp;
    }
    return false;
}

__global__ void findMersennePrimes(unsigned long long *mersennes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long exp = idx + 2; // Starting from 2^2 - 1
        mersennes[idx] = isMersennePrime(pow(2, exp) - 1) ? pow(2, exp) - 1 : 0;
    }
}

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void findCoPrimes(int *coPrimes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int num1 = idx * 100 + rand() % 50;
        int num2 = idx * 100 + rand() % 50;
        coPrimes[idx] = gcd(num1, num2) == 1 ? 1 : 0;
    }
}

__device__ unsigned long long factorial(unsigned long long n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

__global__ void computeFactorials(unsigned long long *factorials, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) factorials[idx] = factorial(idx + 1);
}

__device__ unsigned long long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void computeFibonacci(unsigned long long *fib, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) fib[idx] = fibonacci(idx);
}

__device__ bool isSophieGermainPrime(int num) {
    return isPrime(num) && isPrime(2 * num + 1);
}

__global__ void findSophieGermainPrimes(int *sophiePrimes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) sophiePrimes[idx] = isSophieGermainPrime(idx * 10 + 3) ? idx * 10 + 3 : 0;
}

__device__ unsigned long long carmichaelFunction(unsigned long long n) {
    // Placeholder for actual implementation
    return n;
}

__global__ void computeCarmichael(unsigned long long *carmichaels, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) carmichaels[idx] = carmichaelFunction(idx + 1);
}

__device__ unsigned long long eulerTotientFunction(unsigned long long n) {
    // Placeholder for actual implementation
    return n;
}

__global__ void computeEulerTotient(unsigned long long *eulers, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) eulers[idx] = eulerTotientFunction(idx + 1);
}

int main() {
    srand(time(0));

    const int numPrimes = 1024;
    int *h_primes, *d_primes;
    h_primes = new int[numPrimes];
    cudaMalloc(&d_primes, sizeof(int) * numPrimes);

    generatePrimes<<<(numPrimes + 255) / 256, 256>>>(d_primes, numPrimes);
    cudaMemcpy(h_primes, d_primes, sizeof(int) * numPrimes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numPrimes; ++i) {
        std::cout << "Prime " << i << ": " << h_primes[i] << std::endl;
    }

    cudaFree(d_primes);
    delete[] h_primes;

    return 0;
}
