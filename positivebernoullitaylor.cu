#include <cuda_runtime.h>
#include <cmath>

__device__ unsigned int bernoulliNumber(unsigned int n) {
    if (n == 0) return 1;
    if (n % 2 == 1) return 0;
    double result = 1.0;
    for (unsigned int k = 2; k <= n; k += 2)
        result *= -(k * k - k + 2) / ((k - 1) * (k + 1));
    return static_cast<unsigned int>(result);
}

__device__ unsigned int bernoulliTaylorTerm(unsigned int n, unsigned int x) {
    unsigned int Bn = bernoulliNumber(n);
    if (Bn == 0) return 0;
    double term = pow(x, n + 1) / (n + 1.0);
    return static_cast<unsigned int>(term * Bn);
}

__global__ void generateBernoulliTaylorTerms(unsigned int* terms, unsigned int x, unsigned int numTerms) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        terms[idx] = bernoulliTaylorTerm(idx, x);
}

unsigned int* generateTermsOnDevice(unsigned int x, unsigned int numTerms) {
    unsigned int* d_terms;
    cudaMalloc(&d_terms, numTerms * sizeof(unsigned int));
    generateBernoulliTaylorTerms<<<(numTerms + 255) / 256, 256>>>(d_terms, x, numTerms);
    return d_terms;
}

__device__ bool isPrime(unsigned int n) {
    if (n <= 1) return false;
    for (unsigned int i = 2; i <= sqrt(n); ++i)
        if (n % i == 0) return false;
    return true;
}

__global__ void findPrimesInTerms(unsigned int* terms, unsigned int numTerms, bool* primes) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        primes[idx] = isPrime(terms[idx]);
}

bool* findPrimesOnDevice(unsigned int* terms, unsigned int numTerms) {
    bool* d_primes;
    cudaMalloc(&d_primes, numTerms * sizeof(bool));
    findPrimesInTerms<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_primes);
    return d_primes;
}

__device__ unsigned int sumArrayElements(unsigned int* array, unsigned int length) {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < length; ++i)
        sum += array[i];
    return sum;
}

__global__ void computeSumOfTerms(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedSum[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedSum[threadIdx.x] = terms[idx];
    else
        sharedSum[threadIdx.x] = 0;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicAdd(result, sharedSum[0]);
}

unsigned int sumTermsOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 0;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    computeSumOfTerms<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__device__ unsigned int multiplyArrayElements(unsigned int* array, unsigned int length) {
    unsigned int product = 1;
    for (unsigned int i = 0; i < length; ++i)
        product *= array[i];
    return product;
}

__global__ void computeProductOfTerms(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedProd[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedProd[threadIdx.x] = terms[idx];
    else
        sharedProd[threadIdx.x] = 1;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            sharedProd[threadIdx.x] *= sharedProd[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicMul(result, sharedProd[0]);
}

unsigned int multiplyTermsOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 1;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    computeProductOfTerms<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__device__ unsigned int maxArrayElement(unsigned int* array, unsigned int length) {
    unsigned int maxVal = 0;
    for (unsigned int i = 0; i < length; ++i)
        if (array[i] > maxVal) maxVal = array[i];
    return maxVal;
}

__global__ void findMaxTerm(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedMax[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedMax[threadIdx.x] = terms[idx];
    else
        sharedMax[threadIdx.x] = 0;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s && sharedMax[threadIdx.x] < sharedMax[threadIdx.x + s])
            sharedMax[threadIdx.x] = sharedMax[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicMax(result, sharedMax[0]);
}

unsigned int maxTermOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 0;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    findMaxTerm<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__device__ unsigned int minArrayElement(unsigned int* array, unsigned int length) {
    unsigned int minVal = UINT_MAX;
    for (unsigned int i = 0; i < length; ++i)
        if (array[i] < minVal) minVal = array[i];
    return minVal;
}

__global__ void findMinTerm(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedMin[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedMin[threadIdx.x] = terms[idx];
    else
        sharedMin[threadIdx.x] = UINT_MAX;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s && sharedMin[threadIdx.x] > sharedMin[threadIdx.x + s])
            sharedMin[threadIdx.x] = sharedMin[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicMin(result, sharedMin[0]);
}

unsigned int minTermOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = UINT_MAX;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    findMinTerm<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__device__ unsigned int countPrimesInArray(bool* array, unsigned int length) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < length; ++i)
        if (array[i]) count++;
    return count;
}

__global__ void countPrimes(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedCount[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedCount[threadIdx.x] = isPrime(terms[idx]) ? 1 : 0;
    else
        sharedCount[threadIdx.x] = 0;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            sharedCount[threadIdx.x] += sharedCount[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicAdd(result, sharedCount[0]);
}

unsigned int countPrimesOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 0;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    countPrimes<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__device__ unsigned int isPrime(unsigned int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (unsigned int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    return 1;
}

__global__ void filterPrimes(unsigned int* terms, unsigned int numTerms, bool* isPrimeArray) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        isPrimeArray[idx] = isPrime(terms[idx]);
}

bool* filterPrimesOnDevice(unsigned int* terms, unsigned int numTerms) {
    bool* h_isPrimeArray = new bool[numTerms];
    bool* d_isPrimeArray;
    cudaMalloc(&d_isPrimeArray, sizeof(bool) * numTerms);
    filterPrimes<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_isPrimeArray);
    cudaMemcpy(h_isPrimeArray, d_isPrimeArray, sizeof(bool) * numTerms, cudaMemcpyDeviceToHost);
    cudaFree(d_isPrimeArray);
    return h_isPrimeArray;
}

__global__ void sumPrimes(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedSum[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedSum[threadIdx.x] = isPrime(terms[idx]) ? terms[idx] : 0;
    else
        sharedSum[threadIdx.x] = 0;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicAdd(result, sharedSum[0]);
}

unsigned int sumPrimesOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 0;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    sumPrimes<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__global__ void productPrimes(unsigned int* terms, unsigned int numTerms, unsigned int* result) {
    __shared__ unsigned int sharedProd[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms)
        sharedProd[threadIdx.x] = isPrime(terms[idx]) ? terms[idx] : 1;
    else
        sharedProd[threadIdx.x] = 1;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s)
            sharedProd[threadIdx.x] *= sharedProd[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicMul(result, sharedProd[0]);
}

unsigned int productPrimesOnDevice(unsigned int* terms, unsigned int numTerms) {
    unsigned int h_result = 1;
    unsigned int* d_result;
    cudaMalloc(&d_result, sizeof(unsigned int));
    cudaMemcpy(d_result, &h_result, sizeof(unsigned int), cudaMemcpyHostToDevice);
    productPrimes<<<(numTerms + 255) / 256, 256>>>(terms, numTerms, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}
