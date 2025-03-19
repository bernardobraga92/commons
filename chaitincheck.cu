#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findPrimesChaitinCheck(int* numbers, int size, bool* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        results[idx] = isPrime(numbers[idx]);
    }
}

__global__ void generateRandomNumbers(int* numbers, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        numbers[idx] = curand(&state) % 1000000;
    }
}

__global__ void filterPrimes(int* input, bool* isPrime, int size, int* output, int& count) {
    __shared__ int sharedCount[256];
    if (threadIdx.x < 256) sharedCount[threadIdx.x] = 0;

    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime[i]) {
            atomicAdd(&sharedCount[threadIdx.x], 1);
            output[atomicAdd(&count, 1)] = input[i];
        }
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            atomicAdd(&sharedCount[threadIdx.x], sharedCount[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(&count, sharedCount[0] - count);
}

__global__ void printPrimes(int* primes, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        printf("%d ", primes[i]);
    }
}

__global__ void generateRandomSeeds(unsigned int* seeds, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        seeds[idx] = curand(&state);
    }
}

__global__ void addRandomOffsets(int* numbers, unsigned int* seeds, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        curandState state;
        curand_init(seeds[i], 0, 0, &state);
        numbers[i] += curand(&state) % 1000;
    }
}

__global__ void multiplyByRandomFactors(int* numbers, unsigned int* seeds, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        curandState state;
        curand_init(seeds[i], 0, 0, &state);
        numbers[i] *= (curand(&state) % 10) + 2;
    }
}

__global__ void checkForChaitinPrimes(int* numbers, bool* results, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i]) && isPrime(numbers[i] + 1)) {
            results[i] = true;
        } else {
            results[i] = false;
        }
    }
}

__global__ void reverseArray(int* array, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size / 2; i += blockDim.x * gridDim.x) {
        int temp = array[i];
        array[i] = array[size - i - 1];
        array[size - i - 1] = temp;
    }
}

__global__ void shiftArrayRight(int* array, int size, int shiftAmount) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        array[i] = array[(i - shiftAmount + size) % size];
    }
}

__global__ void shiftArrayLeft(int* array, int size, int shiftAmount) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        array[i] = array[(i + shiftAmount) % size];
    }
}

__global__ void rotateArray(int* array, int size, int rotationAmount) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        array[i] = array[(i + rotationAmount) % size];
    }
}

__global__ void sortArray(int* array, int size) {
    extern __shared__ int shared[];
    for (int s = size / 2; s > 0; s /= 2) {
        for (int j = threadIdx.x; j < size; j += blockDim.x * 2) {
            if ((j + s) < size && array[j] > array[j + s]) {
                int temp = array[j];
                array[j] = array[j + s];
                array[j + s] = temp;
            }
        }
        __syncthreads();
    }
}

__global__ void binarySearch(int* sortedArray, int size, int target) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 1; i += blockDim.x * gridDim.x) {
        int left = 0;
        int right = size - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sortedArray[mid] == target) printf("Found at index %d\n", mid);
            if (sortedArray[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
    }
}

__global__ void findMaxPrime(int* numbers, int size, int& maxPrime) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicMax(&maxPrime, numbers[i]);
    }
}

__global__ void findMinPrime(int* numbers, int size, int& minPrime) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicMin(&minPrime, numbers[i]);
    }
}

__global__ void countPrimes(int* numbers, int size, int& primeCount) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicAdd(&primeCount, 1);
    }
}

__global__ void sumPrimes(int* numbers, int size, int& primeSum) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicAdd(&primeSum, numbers[i]);
    }
}

__global__ void productPrimes(int* numbers, int size, int& primeProduct) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicMul(&primeProduct, numbers[i]);
    }
}

__global__ void gcdOfPrimes(int* numbers, int size, int& primeGCD) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicMin(&primeGCD, numbers[i]);
    }
}

__global__ void lcmOfPrimes(int* numbers, int size, int& primeLCM) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (isPrime(numbers[i])) atomicMax(&primeLCM, numbers[i]);
    }
}

__global__ void sieveOfEratosthenes(bool* primes, int size) {
    for (int p = 2 + threadIdx.x; p < size; p += blockDim.x * gridDim.x) {
        if (primes[p]) {
            for (int i = p * p; i < size; i += p) {
                primes[i] = false;
            }
        }
    }
}

__global__ void eulerTotientFunction(int* numbers, int size) {
    for (int n = 2 + threadIdx.x; n < size; n += blockDim.x * gridDim.x) {
        if (isPrime(numbers[n])) numbers[n - 2] = n - 1;
    }
}

__global__ void carmichaelLambdaFunction(int* numbers, int size) {
    for (int n = 2 + threadIdx.x; n < size; n += blockDim.x * gridDim.x) {
        if (isPrime(numbers[n])) numbers[n - 2] = n - 1;
    }
}

__global__ void legendreSymbol(int* numbers, int size, int p) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, p) == 1) {
            int result = modPow(a, (p - 1) / 2, p);
            numbers[a] = result;
        }
    }
}

__global__ void jacobiSymbol(int* numbers, int size, int n) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, n) == 1) {
            int result = modPow(a, (n - 1) / 2, n);
            numbers[a] = result;
        }
    }
}

__global__ void kroneckerSymbol(int* numbers, int size, int d) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, d) == 1) {
            int result = modPow(a, (d - 1) / 2, d);
            numbers[a] = result;
        }
    }
}

__global__ void quadraticResidue(int* numbers, int size, int p) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, p) == 1) {
            int result = modPow(a, (p - 1) / 2, p);
            numbers[a] = result;
        }
    }
}

__global__ void quadraticNonResidue(int* numbers, int size, int p) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, p) == 1) {
            int result = modPow(a, (p - 1) / 2, p);
            numbers[a] = result;
        }
    }
}

__global__ void legendreSymbol(int* numbers, int size, int p) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, p) == 1) {
            int result = modPow(a, (p - 1) / 2, p);
            numbers[a] = result;
        }
    }
}

__global__ void jacobiSymbol(int* numbers, int size, int n) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, n) == 1) {
            int result = modPow(a, (n - 1) / 2, n);
            numbers[a] = result;
        }
    }
}

__global__ void kroneckerSymbol(int* numbers, int size, int d) {
    for (int a = 0 + threadIdx.x; a < size; a += blockDim.x * gridDim.x) {
        if (gcd(a, d) == 1) {
            int result = modPow(a, (d - 1) / 2, d);
            numbers[a] = result;
        }
    }
}
