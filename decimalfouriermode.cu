#include <iostream>
#include <cmath>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= sqrt(n); i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(idx)) {
            atomicAdd(primes, idx);
        }
    }
}

__device__ int decimalToBinary(int n) {
    int binaryNum = 0;
    int i = 1;
    while (n > 0) {
        binaryNum += (n % 2) * i;
        n /= 2;
        i *= 10;
    }
    return binaryNum;
}

__global__ void decimalFourierMode(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(idx)) {
            atomicAdd(primes, decimalToBinary(idx));
        }
    }
}

__device__ int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void findFibonacciPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int fib = fibonacci(idx);
        if (isPrime(fib)) {
            atomicAdd(primes, fib);
        }
    }
}

__device__ int factorial(int n) {
    return (n == 1 || n == 0) ? 1 : n * factorial(n - 1);
}

__global__ void findFactorialPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int fact = factorial(idx);
        if (isPrime(fact)) {
            atomicAdd(primes, fact);
        }
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

__global__ void findGCDPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(gcd(idx, 10))) {
            atomicAdd(primes, idx);
        }
    }
}

__device__ int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

__global__ void findLCMPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(lcm(idx, 10))) {
            atomicAdd(primes, idx);
        }
    }
}

__device__ int powerMod(int base, int exp, int mod) {
    int result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

__global__ void findPowerModPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(powerMod(idx, 2, 10))) {
            atomicAdd(primes, idx);
        }
    }
}

__device__ int totient(int n) {
    int result = n;
    for (int p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            while (n % p == 0)
                n /= p;
            result -= result / p;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

__global__ void findTotientPrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(totient(idx))) {
            atomicAdd(primes, idx);
        }
    }
}

__device__ bool isPerfectSquare(long long x) {
    long long s = static_cast<long long>(sqrt(x));
    return (s * s == x);
}

__global__ void findPerfectSquarePrimes(int* primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isPrime(idx) && isPerfectSquare(idx)) {
            atomicAdd(primes, idx);
        }
    }
}

int main() {
    const int size = 1000;
    int* h_primes = new int[size];
    memset(h_primes, 0, sizeof(int) * size);

    int* d_primes;
    cudaMalloc(&d_primes, sizeof(int) * size);
    cudaMemcpy(d_primes, h_primes, sizeof(int) * size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    decimalFourierMode<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findFibonacciPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findFactorialPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findGCDPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findLCMPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findPowerModPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findTotientPrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    findPerfectSquarePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);

    cudaMemcpy(h_primes, d_primes, sizeof(int) * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (h_primes[i] != 0) {
            std::cout << h_primes[i] << " ";
        }
    }
    std::cout << std::endl;

    delete[] h_primes;
    cudaFree(d_primes);

    return 0;
}
