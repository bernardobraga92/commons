#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

__device__ int findNextPrime(int start) {
    while (!isPrime(start)) {
        start++;
    }
    return start;
}

__global__ void generatePrimes(int* d_primes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_primes[idx] = findNextPrime(idx * 2 + 1);
    }
}

extern "C" {
    void runGeneratePrimes(int* h_primes, int count) {
        int* d_primes;
        cudaMalloc((void**)&d_primes, count * sizeof(int));
        generatePrimes<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_primes, count);
        cudaMemcpy(h_primes, d_primes, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_primes);
    }
}

__device__ int sigmoid(int x) {
    return 1 / (1 + exp(-x));
}

__global__ void applySigmoidToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = sigmoid(d_data[idx]);
    }
}

extern "C" {
    void runApplySigmoidToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        applySigmoidToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int ordinalgodelsigmoid(int x) {
    return (x * x + x + 41) % 47;
}

__global__ void applyOrdinalgodelsigmoidToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = ordinalgodelsigmoid(d_data[idx]);
    }
}

extern "C" {
    void runApplyOrdinalgodelsigmoidToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        applyOrdinalgodelsigmoidToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int godelNumber(int a, int b) {
    return pow(2, pow(3, a)) * pow(5, b);
}

__global__ void generateGodelNumbersToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = godelNumber(idx % 10, idx / 10);
    }
}

extern "C" {
    void runGenerateGodelNumbersToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        generateGodelNumbersToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int modInverse(int a, int m) {
    for (int x = 1; x < m; x++) {
        if ((a * x) % m == 1) return x;
    }
    return -1;
}

__global__ void computeModInversesToDevice(int* d_data, int count, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = modInverse(d_data[idx], m);
    }
}

extern "C" {
    void runComputeModInversesToDevice(int* h_data, int count, int m) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        computeModInversesToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count, m);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int fermatTest(int num) {
    if (num <= 3) return 1;
    for (int i = 2; i < num; i++) {
        if (pow(i, num - 1) % num != 1) return 0;
    }
    return 1;
}

__global__ void performFermatTestToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = fermatTest(d_data[idx]);
    }
}

extern "C" {
    void runPerformFermatTestToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        performFermatTestToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int eulerTotientFunction(int num) {
    int result = num;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            while (num % i == 0) num /= i;
            result -= result / i;
        }
    }
    if (num > 1) result -= result / num;
    return result;
}

__global__ void computeEulerTotientToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = eulerTotientFunction(d_data[idx]);
    }
}

extern "C" {
    void runComputeEulerTotientToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        computeEulerTotientToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int legendreSymbol(int a, int p) {
    if (a == 0) return 0;
    if (a == 1) return 1;
    if (a % 2 == 0) return legendreSymbol(a / 2, p) * pow(-1, ((p * p - 1) / 8));
    int k = (a - 1) / 2;
    return legendreSymbol(p % a, a) * pow(-1, k * (p * p - 1) / 4);
}

__global__ void computeLegendreSymbolToDevice(int* d_data, int count, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = legendreSymbol(d_data[idx], p);
    }
}

extern "C" {
    void runComputeLegendreSymbolToDevice(int* h_data, int count, int p) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        computeLegendreSymbolToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count, p);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int carmichaelFunction(int num) {
    int result = 1;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            while (num % i == 0) num /= i;
            result = lcm(result, eulerTotientFunction(i));
        }
    }
    if (num > 1) result = lcm(result, eulerTotientFunction(num));
    return result;
}

__global__ void computeCarmichaelFunctionToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = carmichaelFunction(d_data[idx]);
    }
}

extern "C" {
    void runComputeCarmichaelFunctionToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        computeCarmichaelFunctionToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int pollardRhoAlgorithm(int num) {
    if (num % 2 == 0) return 2;
    int x = rand() % num + 1;
    int y = x;
    int c = rand() % num + 1;
    int d = 1;
    while (d == 1) {
        x = (x * x + c) % num;
        y = (y * y + c) % num;
        y = (y * y + c) % num;
        d = gcd(abs(x - y), num);
    }
    return d;
}

__global__ void performPollardRhoToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = pollardRhoAlgorithm(d_data[idx]);
    }
}

extern "C" {
    void runPerformPollardRhoToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        performPollardRhoToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int millerRabinTest(int num) {
    if (num <= 3) return 1;
    for (int a = 2; a < num; a++) {
        if (gcd(a, num) != 1) continue;
        int k = 0, d = num - 1;
        while (d % 2 == 0) d /= 2, k++;
        int x = pow(a, d) % num;
        if (x == 1 || x == num - 1) continue;
        for (int r = 1; r < k; r++) {
            x = (x * x) % num;
            if (x == num - 1) break;
        }
        if (x != num - 1) return 0;
    }
    return 1;
}

__global__ void performMillerRabinTestToDevice(int* d_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_data[idx] = millerRabinTest(d_data[idx]);
    }
}

extern "C" {
    void runPerformMillerRabinTestToDevice(int* h_data, int count) {
        int* d_data;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        performMillerRabinTestToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count);
        cudaMemcpy(h_data, d_data, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

__device__ int extendedEuclideanAlgorithm(int a, int b, int* x, int* y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }
    int x1, y1;
    int gcd = extendedEuclideanAlgorithm(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

__global__ void performExtendedEuclideanToDevice(int* d_data, int count, int* d_x, int* d_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        extendedEuclideanAlgorithm(d_data[idx], 1000, &d_x[idx], &d_y[idx]);
    }
}

extern "C" {
    void runPerformExtendedEuclideanToDevice(int* h_data, int count, int* h_x, int* h_y) {
        int* d_data;
        int* d_x;
        int* d_y;
        cudaMalloc((void**)&d_data, count * sizeof(int));
        cudaMalloc((void**)&d_x, count * sizeof(int));
        cudaMalloc((void**)&d_y, count * sizeof(int));
        cudaMemcpy(d_data, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
        performExtendedEuclideanToDevice<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, count, d_x, d_y);
        cudaMemcpy(h_x, d_x, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_x);
        cudaFree(d_y);
    }
}
