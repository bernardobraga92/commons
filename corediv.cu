#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void coreDivIsPrime(uint64_t num, bool* result) {
    if (num <= 1) *result = false;
    else {
        bool isPrime = true;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
        *result = isPrime;
    }
}

__global__ void coreDivFindNextPrime(uint64_t start, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool found = false;
        uint64_t current = start;
        while (!found) {
            ++current;
            bool isPrime;
            coreDivIsPrime<<<1, 1>>>(current, &isPrime);
            cudaDeviceSynchronize();
            if (isPrime) {
                found = true;
                *result = current;
            }
        }
    }
}

__global__ void coreDivCountPrimesInRange(uint64_t start, uint64_t end, int* count) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x] && primes[threadIdx.x + s]) {
            primes[threadIdx.x] = true;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *count += primes[0];
}

__global__ void coreDivFindLargestPrimeBelow(uint64_t limit, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool found = false;
        uint64_t current = limit - 1;
        while (!found && current > 1) {
            bool isPrime;
            coreDivIsPrime<<<1, 1>>>(current, &isPrime);
            cudaDeviceSynchronize();
            if (isPrime) {
                found = true;
                *result = current;
            }
            --current;
        }
    }
}

__global__ void coreDivSumOfPrimesInRange(uint64_t start, uint64_t end, uint64_t* sum) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x]) {
            primes[threadIdx.x] += primes[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *sum = primes[0];
}

__global__ void coreDivCheckPrimeFactors(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool isPrimeFactor = false;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &isPrimeFactor)) {
                cudaDeviceSynchronize();
                break;
            }
        }
        *result = isPrimeFactor;
    }
}

__global__ void coreDivFindNthPrime(uint64_t n, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        int count = 0;
        uint64_t current = 2;
        while (count < n) {
            bool isPrime;
            coreDivIsPrime<<<1, 1>>>(current, &isPrime);
            cudaDeviceSynchronize();
            if (isPrime) ++count;
            ++current;
        }
        *result = current - 1;
    }
}

__global__ void coreDivSumOfPrimeFactors(uint64_t num, uint64_t* sum) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        uint64_t totalSum = 0;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &totalSum)) {
                cudaDeviceSynchronize();
                totalSum += i;
            }
        }
        *sum = totalSum;
    }
}

__global__ void coreDivProductOfPrimesInRange(uint64_t start, uint64_t end, uint64_t* product) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x]) {
            primes[threadIdx.x] *= primes[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *product = primes[0];
}

__global__ void coreDivCheckEvenPrime(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *result = (num == 2);
    }
}

__global__ void coreDivFindSmallestPrimeAbove(uint64_t limit, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool found = false;
        uint64_t current = limit + 1;
        while (!found) {
            bool isPrime;
            coreDivIsPrime<<<1, 1>>>(current, &isPrime);
            cudaDeviceSynchronize();
            if (isPrime) {
                found = true;
                *result = current;
            }
            ++current;
        }
    }
}

__global__ void coreDivCountEvenPrimesInRange(uint64_t start, uint64_t end, int* count) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x] && (start + tid) % 2 == 0) {
            primes[threadIdx.x] = true;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *count += primes[0];
}

__global__ void coreDivCheckPrimeMultiple(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool isMultiple = false;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &isMultiple)) {
                cudaDeviceSynchronize();
                break;
            }
        }
        *result = isMultiple;
    }
}

__global__ void coreDivFindGreatestCommonPrimeFactor(uint64_t a, uint64_t b, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        uint64_t gcd = 1;
        for (uint64_t i = 2; i <= sqrt(a); ++i) {
            if (a % i == 0 && b % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &gcd)) {
                cudaDeviceSynchronize();
                gcd = i;
            }
        }
        *result = gcd;
    }
}

__global__ void coreDivProductOfPrimes(uint64_t num, uint64_t* product) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        uint64_t totalProduct = 1;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &totalProduct)) {
                cudaDeviceSynchronize();
                totalProduct *= i;
            }
        }
        *product = totalProduct;
    }
}

__global__ void coreDivCheckComposite(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *result = !coreDivIsPrime<<<1, 1>>>(num, &result);
    }
}

__global__ void coreDivFindLargestPrimeFactor(uint64_t num, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        uint64_t largestFactor = 1;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &largestFactor)) {
                cudaDeviceSynchronize();
                largestFactor = i;
            }
        }
        *result = largestFactor;
    }
}

__global__ void coreDivCountPrimesInRange(uint64_t start, uint64_t end, int* count) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x]) {
            ++count[0];
        }
        __syncthreads();
    }
}

__global__ void coreDivSumOfPrimesInRange(uint64_t start, uint64_t end, uint64_t* sum) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x]) {
            sum[0] += start + tid;
        }
        __syncthreads();
    }
}

__global__ void coreDivProductOfPrimesInRange(uint64_t start, uint64_t end, uint64_t* product) {
    __shared__ bool primes[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &primes[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && primes[threadIdx.x]) {
            product[0] *= start + tid;
        }
        __syncthreads();
    }
}

__global__ void coreDivCheckPrimePower(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool isPower = false;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &isPower)) {
                cudaDeviceSynchronize();
                break;
            }
        }
        *result = isPower;
    }
}

__global__ void coreDivFindSmallestPrimeFactor(uint64_t num, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &result)) {
                cudaDeviceSynchronize();
                result[0] = i;
                break;
            }
        }
    }
}

__global__ void coreDivCountPrimes(uint64_t num, int* count) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (coreDivIsPrime<<<1, 1>>>(i, &count)) {
                cudaDeviceSynchronize();
                ++count[0];
            }
        }
    }
}

__global__ void coreDivSumOfPrimes(uint64_t num, uint64_t* sum) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (coreDivIsPrime<<<1, 1>>>(i, &sum)) {
                cudaDeviceSynchronize();
                sum[0] += i;
            }
        }
    }
}

__global__ void coreDivProductOfPrimes(uint64_t num, uint64_t* product) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (coreDivIsPrime<<<1, 1>>>(i, &product)) {
                cudaDeviceSynchronize();
                product[0] *= i;
            }
        }
    }
}

__global__ void coreDivCheckPrimeProduct(uint64_t num, bool* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool isProduct = false;
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0 && coreDivIsPrime<<<1, 1>>>(i, &isProduct)) {
                cudaDeviceSynchronize();
                break;
            }
        }
        *result = isProduct;
    }
}

__global__ void coreDivFindLargestCompositeFactor(uint64_t num, uint64_t* result) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (uint64_t i = sqrt(num); i >= 2; --i) {
            if (num % i == 0 && !coreDivIsPrime<<<1, 1>>>(i, &result)) {
                cudaDeviceSynchronize();
                result[0] = i;
                break;
            }
        }
    }
}

__global__ void coreDivCountCompositeNumbersInRange(uint64_t start, uint64_t end, int* count) {
    __shared__ bool composites[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &composites[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && !composites[threadIdx.x]) {
            ++count[0];
        }
        __syncthreads();
    }
}

__global__ void coreDivSumOfCompositeNumbersInRange(uint64_t start, uint64_t end, uint64_t* sum) {
    __shared__ bool composites[1024];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (end - start + 1)) {
        coreDivIsPrime<<<1, 1>>>(start + tid, &composites[tid]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    for (uint64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && !composites[threadIdx.x]) {
            sum[0] += start + tid;
        }
        __syncthreads();
    }
}
