#include <stdio.h>
#include <stdlib.h>

__global__ void cardinalAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void descartesProduct(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void vecAddModulo(int* a, int* b, int* c, int n, int mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (a[idx] + b[idx]) % mod;
}

__global__ void cardinalMultiply(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void descartesSubtract(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void vecSubtractModulo(int* a, int* b, int* c, int n, int mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (a[idx] - b[idx] + mod) % mod;
}

__global__ void cardinalDivide(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && b[idx] != 0) c[idx] = a[idx] / b[idx];
}

__global__ void descartesPower(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = pow(a[idx], b[idx]);
}

__global__ void vecMultiplyModulo(int* a, int* b, int* c, int n, int mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (a[idx] * b[idx]) % mod;
}

__global__ void cardinalModulo(int* a, int* b, int* c, int n, int mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] % mod;
}

__global__ void descartesGCD(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = gcd(a[idx], b[idx]);
}

__global__ void vecAddPrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime(a[idx] + b[idx]);
}

__global__ void cardinalSubtractPrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime(a[idx] - b[idx]);
}

__global__ void descartesMultiplyPrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime(a[idx] * b[idx]);
}

__global__ void vecDividePrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && b[idx] != 0) c[idx] = isPrime(a[idx] / b[idx]);
}

__global__ void cardinalPowerPrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime(pow(a[idx], b[idx]));
}

__global__ void descartesModuloPrimeCheck(int* a, int* b, int* c, int n, int mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime((a[idx] % mod));
}

__global__ void vecGCDPrimeCheck(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = isPrime(gcd(a[idx], b[idx]));
}
