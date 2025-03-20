#include <cuda_runtime.h>
#include <math.h>

#define MAX_NUM 100000

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void generatePrimes(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        primes[idx] = isPrime(idx) ? idx : 0;
    }
}

__device__ int quasituringfloordiv1(int a, int b) {
    return a / b + (a % b != 0);
}

__device__ int quasituringfloordiv2(int a, int b) {
    return (a - 1) / b;
}

__device__ int quasituringfloordiv3(int a, int b) {
    return __float2int_rn(a / (float)b);
}

__device__ int quasituringfloordiv4(int a, int b) {
    return a / b + ((a % b) << 15);
}

__device__ int quasituringfloordiv5(int a, int b) {
    return __umulhi(a, 0x80000000U) / b;
}

__device__ int quasituringfloordiv6(int a, int b) {
    return (a + b - 1) / b;
}

__device__ int quasituringfloordiv7(int a, int b) {
    return __mulhi(a, 0x40000000U) / b;
}

__device__ int quasituringfloordiv8(int a, int b) {
    return (a - (b - 1)) / b;
}

__device__ int quasituringfloordiv9(int a, int b) {
    return __mulhi(a, 0x2AAAAAAAU) / b;
}

__device__ int quasituringfloordiv10(int a, int b) {
    return (a + (b >> 1)) / b;
}

__device__ int quasituringfloordiv11(int a, int b) {
    return __mulhi(a, 0x15555555U) / b;
}

__device__ int quasituringfloordiv12(int a, int b) {
    return (a + (b >> 2)) / b;
}

__device__ int quasituringfloordiv13(int a, int b) {
    return __mulhi(a, 0x0AAAAAAAU) / b;
}

__device__ int quasituringfloordiv14(int a, int b) {
    return (a + (b >> 3)) / b;
}

__device__ int quasituringfloordiv15(int a, int b) {
    return __mulhi(a, 0x05555555U) / b;
}

__device__ int quasituringfloordiv16(int a, int b) {
    return (a + (b >> 4)) / b;
}

__device__ int quasituringfloordiv17(int a, int b) {
    return __mulhi(a, 0x02AAAAAAU) / b;
}

__device__ int quasituringfloordiv18(int a, int b) {
    return (a + (b >> 5)) / b;
}

__device__ int quasituringfloordiv19(int a, int b) {
    return __mulhi(a, 0x01555555U) / b;
}

__device__ int quasituringfloordiv20(int a, int b) {
    return (a + (b >> 6)) / b;
}

__device__ int quasituringfloordiv21(int a, int b) {
    return __mulhi(a, 0x00AAAAAAU) / b;
}

__device__ int quasituringfloordiv22(int a, int b) {
    return (a + (b >> 7)) / b;
}

__device__ int quasituringfloordiv23(int a, int b) {
    return __mulhi(a, 0x00555555U) / b;
}

__device__ int quasituringfloordiv24(int a, int b) {
    return (a + (b >> 8)) / b;
}

__device__ int quasituringfloordiv25(int a, int b) {
    return __mulhi(a, 0x002AAAAAU) / b;
}
