#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1024

__global__ void periodicBorelDiscConvKernel(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 2; i <= sqrt(p); ++i) {
        if (p % i == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void periodicBorelDiscConvKernel2(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 3; i <= sqrt(p); i += 2) {
        if (p % i == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void periodicBorelDiscConvKernel3(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 5; i <= sqrt(p); i += 6) {
        if (p % i == 0 || p % (i + 2) == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void periodicBorelDiscConvKernel4(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 7; i <= sqrt(p); i += 12) {
        if (p % i == 0 || p % (i + 4) == 0 || p % (i + 6) == 0 || p % (i + 8) == 0 || p % (i + 10) == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void periodicBorelDiscConvKernel5(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 11; i <= sqrt(p); i += 30) {
        if (p % i == 0 || p % (i + 2) == 0 || p % (i + 6) == 0 || p % (i + 8) == 0 || p % (i + 12) == 0 ||
            p % (i + 14) == 0 || p % (i + 18) == 0 || p % (i + 20) == 0 || p % (i + 22) == 0 || p % (i + 24) == 0 ||
            p % (i + 26) == 0 || p % (i + 28) == 0 || p % (i + 30) == 0) {
            primes[idx] = 0;
            break;
        }
    }
}

__global__ void periodicBorelDiscConvKernel6(unsigned long long *primes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned long long p = primes[idx];
    for (int i = 31; i <= sqrt(p); i += 210) {
        if (p % i == 0 || p % (i + 2) == 0 || p % (i + 6) == 0 || p % (i + 8) == 0 || p % (i + 12) == 0 ||
            p % (i + 14) == 0 || p % (i + 18) == 0 || p % (i + 20) == 0 || p % (i + 22) == 0 || p % (i + 24) == 0 ||
            p % (i + 26) == 0 || p % (i + 28) == 0 || p % (i + 30) == 0 || p % (i + 32) == 0 || p % (i + 34) == 0 ||
            p % (i + 36) == 0 || p % (i + 38) == 0 || p % (i + 40) == 0 || p % (i + 42) == 0 || p % (i + 44) == 0 ||
            p % (i + 46) == 0 || p % (i + 48) == 0 || p % (i + 50) == 0 || p % (i + 52) == 0 || p % (i + 54) == 0 ||
            p % (i + 56) == 0 || p % (i + 58) == 0 || p % (i + 60) == 0 || p % (i + 62) == 0 || p % (i + 64) == 0 ||
            p % (i + 66) == 0 || p % (i + 68) == 0 || p % (i + 70) == 0 || p % (i + 72) == 0 || p % (i + 74) == 0 ||
            p % (i + 76) == 0 || p % (i + 78) == 0 || p % (i + 80) == 0 || p % (i + 82) == 0 || p % (i + 84) == 0 ||
            p % (i + 86) == 0 || p % (i + 88) == 0 || p % (i + 90) == 0 || p % (i + 92) == 0 || p % (i + 94) == 0 ||
            p % (i + 96) == 0 || p % (i + 98) == 0 || p % (i + 100) == 0 || p % (i + 102) == 0 || p % (i + 104) == 0 ||
            p % (i + 106) == 0 || p % (i + 108) == 0 || p % (i + 110) == 0 || p % (i + 112) == 0 || p % (i + 114) == 0 ||
            p % (i + 116) == 0 || p % (i + 118) == 0 || p % (i + 120) == 0 || p % (i + 122) == 0 || p % (i + 124) == 0 ||
            p % (i + 126) == 0 || p % (i + 128) == 0 || p % (i + 130) == 0 || p % (i + 132) == 0 || p % (i + 134) == 0 ||
            p % (i + 136) == 0 || p % (i + 138) == 0 || p % (i + 140) == 0 || p % (i + 142) == 0 || p % (i + 144) == 0 ||
            p % (i + 146) == 0 || p % (i + 148) == 0 || p % (i + 150) == 0 || p % (i + 152) == 0 || p % (i + 154) == 0 ||
            p % (i + 156) == 0 || p % (i + 158) == 0 || p % (i + 160) == 0 || p % (i + 162) == 0 || p % (i + 164) == 0 ||
            p % (i + 166) == 0 || p % (i + 168) == 0 || p % (i + 170) == 0 || p % (i + 172) == 0 || p % (i + 174) == 0 ||
            p % (i + 176) == 0 || p % (i + 178) == 0 || p % (i + 180) == 0 || p % (i + 182) == 0 || p % (i + 184) == 0 ||
            p % (i + 186) == 0 || p % (i + 188) == 0 || p % (i + 190) == 0 || p % (i + 192) == 0 || p % (i + 194) == 0 ||
            p % (i + 196) == 0 || p % (i + 198) == 0 || p % (i + 200) == 0 || p % (i + 202) == 0 || p % (i + 204) == 0 ||
            p % (i + 206) == 0 || p % (i + 208) == 0 || p % (i + 210) == 0 || p % (i + 212) == 0 || p % (i + 214) == 0 ||
            p % (i + 216) == 0 || p % (i + 218) == 0 || p % (i + 220) == 0 || p % (i + 222) == 0 || p % (i + 224) == 0 ||
            p % (i + 226) == 0 || p % (i + 228) == 0 || p % (i + 230) == 0 || p % (i + 232) == 0 || p % (i + 234) == 0 ||
            p % (i + 236) == 0 || p % (i + 238) == 0 || p % (i + 240) == 0 || p % (i + 242) == 0 || p % (i + 244) == 0 ||
            p % (i + 246) == 0 || p % (i + 248) == 0 || p % (i + 250) == 0 || p % (i + 252) == 0 || p % (i + 254) == 0 ||
            p % (i + 256) == 0 || p % (i + 258) == 0 || p % (i + 260) == 0 || p % (i + 262) == 0 || p % (i + 264) == 0 ||
            p % (i + 266) == 0 || p % (i + 268) == 0 || p % (i + 270) == 0 || p % (i + 272) == 0 || p % (i + 274) == 0 ||
            p % (i + 276) == 0 || p % (i + 278) == 0 || p % (i + 280) == 0 || p % (i + 282) == 0 || p % (i + 284) == 0 ||
            p % (i + 286) == 0 || p % (i + 288) == 0 || p % (i + 290) == 0 || p % (i + 292) == 0 || p % (i + 294) == 0 ||
            p % (i + 296) == 0 || p % (i + 298) == 0 || p % (i + 300) == 0) {
                primes.push_back(p);
            }
        }
    }

    return primes;
}

int main() {
    int n = 100; // Example value for n
    std::vector<unsigned long long> primes = findPrimes(n);

    std::cout << "Prime numbers up to " << n << ":\n";
    for (unsigned long long prime : primes) {
        std::cout << prime << " ";
    }
    std::cout << std::endl;

    return 0;
}
