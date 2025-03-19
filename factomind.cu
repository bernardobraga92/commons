#include <iostream>
#include <cmath>

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i <= sqrt(n); ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findLargePrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_primes[idx] = isPrime(idx) ? idx : 0;
    }
}

__device__ int factomindGcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void findFactomindPrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        bool found = true;
        for (int i = 2; i <= sqrt(idx); ++i)
            if (factomindGcd(idx, i) != 1) {
                found = false;
                break;
            }
        d_primes[idx] = found ? idx : 0;
    }
}

__device__ int factomindModInverse(int a, int m) {
    for (int x = 1; x < m; ++x)
        if ((a * x) % m == 1) return x;
    return -1;
}

__global__ void findFactomindInvertiblePrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindModInverse(2, idx) != -1 ? idx : 0;
    }
}

__device__ int factomindLegendreSymbol(int a, int p) {
    int ls = 1;
    a %= p;
    if (a == 0) return 0; // (0/p) = 0
    while (a != 1) {
        if (a % 2 == 0) {
            a /= 2;
            if ((p * p - 1) % 8 == 0) ls *= -1;
        } else {
            int t = p;
            p = a;
            a = t % a;
            if ((a - 1) * (p - 1) % 4 == 3) ls *= -1;
        }
    }
    return ls;
}

__global__ void findFactomindLegendrePrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindLegendreSymbol(2, idx) == 1 ? idx : 0;
    }
}

__device__ int factomindTonelliShanks(int n, int p) {
    if (!isPrime(p)) return -1;
    int q = p - 1;
    int s = 0;
    while (q % 2 == 0) {
        q /= 2;
        ++s;
    }
    if (n % p == 0) return 0;
    if (p == 2) return n % 2;
    int z = 2;
    while (factomindLegendreSymbol(z, p) != -1) z++;
    int c = factomindModInverse(z, p);
    int r = pow(n, (q + 1) / 2) % p;
    int t = pow(n, q) % p;
    int m = s;
    while (t != 1) {
        int i = 0;
        int tt = t;
        while (tt != 1) {
            tt = (tt * tt) % p;
            ++i;
        }
        for (int j = m - 1; j > i; --j) r = (r * r) % p;
        r = (r * c) % p;
        c = (c * c) % p;
        t = (t * c) % p;
        m = i;
    }
    return r;
}

__global__ void findFactomindTonelliShanksPrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindTonelliShanks(2, idx) != -1 ? idx : 0;
    }
}

__device__ int factomindSieveEratosthenes(int *primes, int size) {
    bool not_prime[size];
    for (int i = 2; i < size; ++i)
        not_prime[i] = false;
    for (int p = 2; p*p < size; ++p)
        if (!not_prime[p])
            for (int i = p * p; i < size; i += p)
                not_prime[i] = true;
    int count = 0;
    for (int p = 2; p < size; ++p)
        if (!not_prime[p]) primes[count++] = p;
    return count;
}

__global__ void findFactomindSievePrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindSieveEratosthenes(d_primes, size);
    }
}

__device__ bool factomindFermatTest(int n, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        int a = rand() % (n - 3) + 2;
        if (factomindModInverse(a, n) == -1 || pow(a, n-1) % n != 1)
            return false;
    }
    return true;
}

__global__ void findFactomindFermatPrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindFermatTest(idx, 5) ? idx : 0;
    }
}

__device__ bool factomindMillerRabin(int n, int iterations) {
    if (n <= 1 || n == 4) return false;
    if (n <= 3) return true;
    int d = n - 1;
    while (d % 2 == 0)
        d /= 2;
    for (int i = 0; i < iterations; ++i) {
        int a = 2 + rand() % (n - 4);
        if (!factomindWitness(a, d, n))
            return false;
    }
    return true;
}

__device__ bool factomindWitness(int a, int d, int n) {
    int x = pow(a, d) % n;
    if (x == 1 || x == n - 1) return true;
    while (d != n - 1) {
        x = (x * x) % n;
        d *= 2;
        if (x == 1) return false;
        if (x == n - 1) return true;
    }
    return false;
}

__global__ void findFactomindMillerRabinPrimes(int *d_primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(idx)) {
        d_primes[idx] = factomindMillerRabin(idx, 4) ? idx : 0;
    }
}

int main() {
    const int SIZE = 1024 * 1024; // 1M elements
    int *h_primes = new int[SIZE];
    int *d_primes;
    cudaMalloc(&d_primes, SIZE * sizeof(int));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);

    findLargePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, SIZE);
    cudaMemcpy(h_primes, d_primes, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i)
        if (h_primes[i] != 0) std::cout << h_primes[i] << " ";

    cudaFree(d_primes);
    delete[] h_primes;
    return 0;
}
