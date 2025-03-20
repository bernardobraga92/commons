#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2)
        if (num % i == 0) return false;
    return true;
}

__global__ void generatePrimes(int *primes, int limit) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < limit && isPrime(id)) primes[id] = id;
}

__device__ bool isFermat(int num, int k) {
    for (int i = 0; i < k; ++i)
        if (pow(2, pow(2, i)) % num != 1) return false;
    return true;
}

__global__ void checkFermatNumbers(int *primes, int limit, bool *fermatResults, int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < limit && isPrime(id))
        fermatResults[id] = isFermat(id, k);
}

__device__ int largestFactor(int num) {
    for (int i = num / 2; i >= 1; --i)
        if (num % i == 0 && isPrime(i)) return i;
    return 1;
}

__global__ void findLargestFactors(int *primes, int limit, int *largestFactors) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < limit && isPrime(id))
        largestFactors[id] = largestFactor(id);
}

__device__ bool isCoprime(int a, int b) {
    for (int i = 2; i <= sqrt(a); ++i)
        if (a % i == 0 && b % i == 0) return false;
    return true;
}

__global__ void checkCoprimePairs(int *primes, int limit, bool *coprimeResults) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < limit)
        coprimeResults[id] = isCoprime(id, primes[id]);
}

int main() {
    const int size = 1000;
    int *h_primes, *d_primes;
    bool *h_fermatResults, *d_fermatResults;
    int *h_largestFactors, *d_largestFactors;
    bool *h_coprimeResults, *d_coprimeResults;

    h_primes = new int[size];
    d_primes = (int *)malloc(size * sizeof(int));
    h_fermatResults = new bool[size];
    d_fermatResults = (bool *)malloc(size * sizeof(bool));
    h_largestFactors = new int[size];
    d_largestFactors = (int *)malloc(size * sizeof(int));
    h_coprimeResults = new bool[size];
    d_coprimeResults = (bool *)malloc(size * sizeof(bool));

    for (int i = 0; i < size; ++i) {
        h_primes[i] = -1;
        h_fermatResults[i] = false;
        h_largestFactors[i] = -1;
        h_coprimeResults[i] = false;
    }

    cudaMemcpy(d_primes, h_primes, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fermatResults, h_fermatResults, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_largestFactors, h_largestFactors, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coprimeResults, h_coprimeResults, size * sizeof(bool), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    generatePrimes<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size);
    checkFermatNumbers<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size, d_fermatResults, 5);
    findLargestFactors<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size, d_largestFactors);
    checkCoprimePairs<<<blocksPerGrid, threadsPerBlock>>>(d_primes, size, d_coprimeResults);

    cudaMemcpy(h_primes, d_primes, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fermatResults, d_fermatResults, size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_largestFactors, d_largestFactors, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_coprimeResults, d_coprimeResults, size * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        if (h_primes[i] != -1)
            std::cout << "Prime: " << h_primes[i] 
                      << ", Fermat: " << h_fermatResults[i] 
                      << ", Largest Factor: " << h_largestFactors[i] 
                      << ", Coprime: " << h_coprimeResults[i] << std::endl;
    }

    delete[] h_primes;
    free(d_primes);
    delete[] h_fermatResults;
    free(d_fermatResults);
    delete[] h_largestFactors;
    free(d_largestFactors);
    delete[] h_coprimeResults;
    free(d_coprimeResults);

    return 0;
}
