#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimes<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesInRange(int *primes, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx <= end && isPrime(idx)) {
        primes[idx - start] = idx;
    } else {
        primes[idx - start] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int start, int end) {
    findPrimesInRange<<<(end - start + 255) / 256, 256>>>(primes, start, end);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithOffset(int *primes, int offset, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx > 1 && isPrime(idx)) {
        primes[idx - offset] = idx;
    } else {
        primes[idx - offset] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int offset, int limit) {
    findPrimesWithOffset<<<(limit + 255) / 256, 256>>>(primes, offset, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithStep(int *primes, int start, int step, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x * step + start;
    if (idx > 1 && isPrime(idx)) {
        primes[blockIdx.x * blockDim.x + threadIdx.x] = idx;
    } else {
        primes[blockIdx.x * blockDim.x + threadIdx.x] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int start, int step, int count) {
    findPrimesWithStep<<<(count + 255) / 256, 256>>>(primes, start, step, count);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithSkip(int *primes, int start, int skip, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x * skip + start;
    if (idx > 1 && isPrime(idx)) {
        primes[idx - start] = idx;
    } else {
        primes[idx - start] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int start, int skip, int limit) {
    findPrimesWithSkip<<<(limit + 255) / 256, 256>>>(primes, start, skip, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithExponent(int *primes, int base, int exp, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = pow(base, idx);
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int base, int exp, int limit) {
    findPrimesWithExponent<<<(limit + 255) / 256, 256>>>(primes, base, exp, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithFactorial(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && isPrime(idx)) {
        primes[idx] = idx;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithFactorial<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithProduct(int *primes, int a, int b, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = a * b * idx;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int a, int b, int limit) {
    findPrimesWithProduct<<<(limit + 255) / 256, 256>>>(primes, a, b, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithSum(int *primes, int a, int b, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = a + b + idx;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int a, int b, int limit) {
    findPrimesWithSum<<<(limit + 255) / 256, 256>>>(primes, a, b, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithDifference(int *primes, int a, int b, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = a - b + idx;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int a, int b, int limit) {
    findPrimesWithDifference<<<(limit + 255) / 256, 256>>>(primes, a, b, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithQuotient(int *primes, int a, int b, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = a / b + idx;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int a, int b, int limit) {
    findPrimesWithQuotient<<<(limit + 255) / 256, 256>>>(primes, a, b, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithModulo(int *primes, int a, int b, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = a % b + idx;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int a, int b, int limit) {
    findPrimesWithModulo<<<(limit + 255) / 256, 256>>>(primes, a, b, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithPowerModulo(int *primes, int base, int exp, int mod, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = pow(base, idx) % mod;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int base, int exp, int mod, int limit) {
    findPrimesWithPowerModulo<<<(limit + 255) / 256, 256>>>(primes, base, exp, mod, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithFibonacci(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = 0, b = 1, num;
    for (int i = 2; i <= idx; ++i) {
        num = a + b;
        a = b;
        b = num;
    }
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithFibonacci<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithLucas(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = 2, b = 1, num;
    for (int i = 2; i <= idx; ++i) {
        num = a + b;
        a = b;
        b = num;
    }
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithLucas<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithPell(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = 1, b = 2, num;
    for (int i = 2; i <= idx; ++i) {
        num = 2 * a + b;
        a = b;
        b = num;
    }
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithPell<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithMersenne(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = (1 << idx) - 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithMersenne<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithFermat(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = (1 << (1 << idx)) + 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithFermat<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithEuler(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * idx - idx + 41;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithEuler<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithCullen(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * (1 << idx) + 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithCullen<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithWoodall(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * (1 << idx) - 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithWoodall<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithRiesel(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * (1 << idx) - 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithRiesel<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithSierpinski(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = idx * idx + idx + 41;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithSierpinski<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithThabit(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = 3 * (1 << idx) - 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithThabit<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithFermat(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = (1 << (1 << idx)) + 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithFermat<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithMersenne(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = (1 << idx) - 1;
    if (num > 1 && isPrime(num)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}

extern "C" void oddpowerborelabs(int *primes, int limit) {
    findPrimesWithMersenne<<<(limit + 255) / 256, 256>>>(primes, limit);
}


#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesWithSophieGermain(int *primes, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = 2 * idx + 1;
    if (num > 1 && isPrime(num) && isPrime(2 * num + 1)) {
        primes[idx] = num;
    } else {
        primes[idx] = 0;
    }
}
