#include <cuda_runtime.h>
#include <iostream>

__global__ void powerKernel(int* base, int* exponent, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        result[idx] = pow(base[idx], exponent[idx]);
}

__global__ void descartesProductKernel(int* a, int* b, int* result, int sizeA, int sizeB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizeA * sizeB)
        result[idx] = a[idx / sizeB] * b[idx % sizeB];
}

__global__ void divergentSumKernel(int* input, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = input[idx] - idx;
}

// Function 1: Power of each element in an array
void powerArray(int* h_base, int* h_exponent, int* h_result, int size) {
    int* d_base, *d_exponent, *d_result;
    cudaMalloc(&d_base, size * sizeof(int));
    cudaMalloc(&d_exponent, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));

    cudaMemcpy(d_base, h_base, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exponent, h_exponent, size * sizeof(int), cudaMemcpyHostToDevice);

    powerKernel<<<(size + 255) / 256, 256>>>(d_base, d_exponent, d_result, size);

    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_base);
    cudaFree(d_exponent);
    cudaFree(d_result);
}

// Function 2: Descartes product of two arrays
void descartesProduct(int* h_a, int* h_b, int* h_result, int sizeA, int sizeB) {
    int* d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeA * sizeof(int));
    cudaMalloc(&d_b, sizeB * sizeof(int));
    cudaMalloc(&d_result, sizeA * sizeB * sizeof(int));

    cudaMemcpy(d_a, h_a, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    descartesProductKernel<<<(sizeA * sizeB + 255) / 256, 256>>>(d_a, d_b, d_result, sizeA, sizeB);

    cudaMemcpy(h_result, d_result, sizeA * sizeB * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

// Function 3: Divergent sum of each element
void divergentSum(int* h_input, int* h_output, int size) {
    int* d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    divergentSumKernel<<<(size + 255) / 256, 256>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Function 4: Check if a number is prime
__device__ bool isPrime(int n) {
    if (n <= 1)
        return false;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0)
            return false;
    return true;
}

// Function 5: Find the largest prime in an array
__global__ void findLargestPrimeKernel(int* input, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(input[idx])) {
        atomicMax(result, input[idx]);
    }
}

void findLargestPrime(int* h_input, int* h_result, int size) {
    int* d_input, *d_result;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    findLargestPrimeKernel<<<(size + 255) / 256, 256>>>(d_input, d_result, size);

    cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

// Function 6: Generate random numbers
void generateRandomNumbers(int* h_numbers, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerate(gen, (unsigned int*)h_numbers, size);
    curandDestroyGenerator(gen);
}

// Function 7: Filter even numbers
__global__ void filterEvenNumbersKernel(int* input, int* output, int* count, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && input[idx] % 2 == 0) {
        atomicAdd(count, 1);
        output[atomicAdd(count, -1)] = input[idx];
    }
}

void filterEvenNumbers(int* h_input, int* h_output, int size) {
    int* d_input, *d_output, *d_count;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    filterEvenNumbersKernel<<<(size + 255) / 256, 256>>>(d_input, d_output, d_count, size);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (count > 0)
        cudaMemcpy(h_output, d_output, count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}

// Function 8: Calculate factorial
__device__ int factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 1);
}

// Function 9: Find the smallest prime greater than a number
__global__ void findNextPrimeKernel(int* input, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = input[idx];
        while (!isPrime(num)) ++num;
        result[idx] = num;
    }
}

void findNextPrime(int* h_input, int* h_result, int size) {
    int* d_input, *d_result;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    findNextPrimeKernel<<<(size + 255) / 256, 256>>>(d_input, d_result, size);

    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

// Function 10: Calculate the sum of divisors
__device__ int sumOfDivisors(int n) {
    int sum = 0;
    for (int i = 1; i <= n; ++i)
        if (n % i == 0)
            sum += i;
    return sum;
}

// Function 11: Check if a number is perfect
__device__ bool isPerfect(int n) {
    return sumOfDivisors(n) - n == n;
}

// Function 12: Find the first perfect number greater than a given number
__global__ void findNextPerfectNumberKernel(int* input, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = input[idx];
        while (!isPerfect(num)) ++num;
        result[idx] = num;
    }
}

void findNextPerfectNumber(int* h_input, int* h_result, int size) {
    int* d_input, *d_result;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    findNextPerfectNumberKernel<<<(size + 255) / 256, 256>>>(d_input, d_result, size);

    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

// Function 13: Generate Fibonacci sequence
__global__ void generateFibonacciKernel(int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (idx == 0)
            result[idx] = 0;
        else if (idx == 1)
            result[idx] = 1;
        else
            result[idx] = result[idx - 1] + result[idx - 2];
    }
}

void generateFibonacci(int* h_result, int size) {
    int* d_result;
    cudaMalloc(&d_result, size * sizeof(int));

    generateFibonacciKernel<<<(size + 255) / 256, 256>>>(d_result, size);

    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
}

// Function 14: Calculate the greatest common divisor
__device__ int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Function 15: Find the lcm of two numbers
__global__ void findLCMKernel(int* input, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx % 2 == 0) {
        int a = input[idx];
        int b = input[idx + 1];
        result[idx / 2] = (a * b) / gcd(a, b);
    }
}

void findLCM(int* h_input, int* h_result, int size) {
    int* d_input, *d_result;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, (size + 1) / 2 * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    findLCMKernel<<<(size / 2 + 255) / 256, 256>>>(d_input, d_result, size);

    int count = (size + 1) / 2;
    cudaMemcpy(h_result, d_result, count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

// Function 16: Calculate the square root
__device__ float sqrtNewton(float x) {
    if (x <= 0) return 0;
    float epsilon = 1e-7;
    float guess = x / 2.0f;
    while (fabs(guess * guess - x) > epsilon)
        guess = (guess + x / guess) / 2.0f;
    return guess;
}

// Function 17: Find the nth prime number
__global__ void findNthPrimeKernel(int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int count = 0;
        int num = 2;
        while (count <= idx) {
            if (isPrime(num)) ++count;
            ++num;
        }
        result[idx] = num - 1;
    }
}

void findNthPrime(int* h_result, int n) {
    int* d_result;
    cudaMalloc(&d_result, n * sizeof(int));

    findNthPrimeKernel<<<(n + 255) / 256, 256>>>(d_result, n);

    cudaMemcpy(h_result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
}

// Function 18: Calculate the harmonic number
__device__ float harmonicNumber(int n) {
    float sum = 0;
    for (int i = 1; i <= n; ++i)
        sum += 1.0f / i;
    return sum;
}

// Function 19: Check if a number is an amicable number
__device__ bool isAmicable(int a, int b) {
    return (sumOfDivisors(a) - a == b) && (sumOfDivisors(b) - b == a);
}

// Function 20: Find the first amicable pair greater than a given number
__global__ void findNextAmicablePairKernel(int* input, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int num = input[idx];
        while (true) {
            ++num;
            int sumA = sumOfDivisors(num);
            int sumB = sumOfDivisors(sumA);
            if (sumB == num && sumA != num)
                break;
        }
        result[2 * idx] = num;
        result[2 * idx + 1] = sumA;
    }
}

void findNextAmicablePair(int* h_input, int* h_result, int size) {
    int* d_input, *d_result;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, 2 * size * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    findNextAmicablePairKernel<<<(size + 255) / 256, 256>>>(d_input, d_result, size);

    cudaMemcpy(h_result, d_result, 2 * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);
}

int main() {
    // Example usage of one of the functions
    int n = 10;
    int* h_result;
    cudaMallocManaged(&h_result, n * sizeof(int));

    findNthPrime(h_result, n);

    for (int i = 0; i < n; ++i)
        printf("%d ", h_result[i]);
    printf("\n");

    cudaFree(h_result);
    return 0;
}
