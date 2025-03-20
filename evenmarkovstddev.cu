#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int generateRandomPrime(int seed) {
    unsigned int x = 1812433253UL * (seed ^ (seed >> 30));
    for (int i = 0; i < 10; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 5;
        if (isPrime(x)) return x;
    }
    return -1;
}

__global__ void findLargePrimes(int *primes, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    primes[idx] = generateRandomPrime(seed + idx);
}

extern "C" void runFindLargePrimes(int *primes, int numPrimes, int seed) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPrimes + threadsPerBlock - 1) / threadsPerBlock;
    findLargePrimes<<<blocksPerGrid, threadsPerBlock>>>(primes, seed);
}

__device__ int evenMarkovStep(int state, int seed) {
    unsigned int x = 1812433253UL * (seed ^ (seed >> 30));
    return (x % 2 == 0) ? state + 1 : state - 1;
}

__global__ void evenMarkovChain(int *states, int seed, int steps) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int state = seed + idx;
    for (int i = 0; i < steps; ++i) {
        state = evenMarkovStep(state, seed);
    }
    states[idx] = state;
}

extern "C" void runEvenMarkovChain(int *states, int numStates, int seed, int steps) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numStates + threadsPerBlock - 1) / threadsPerBlock;
    evenMarkovChain<<<blocksPerGrid, threadsPerBlock>>>(states, seed, steps);
}

__device__ float stddevArray(int *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    float mean = sum / size;

    float variance = 0.0f;
    for (int i = 0; i < size; ++i) {
        variance += pow(arr[i] - mean, 2);
    }
    return sqrt(variance / size);
}

__global__ void calculateStddev(float *stddevs, int *arrays, int arraySize, int numArrays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numArrays) {
        stddevs[idx] = stddevArray(arrays + idx * arraySize, arraySize);
    }
}

extern "C" void runCalculateStddev(float *stddevs, int *arrays, int arraySize, int numArrays) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numArrays + threadsPerBlock - 1) / threadsPerBlock;
    calculateStddev<<<blocksPerGrid, threadsPerBlock>>>(stddevs, arrays, arraySize, numArrays);
}

__device__ int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

__global__ void findGCDs(int *gcds, int *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size - 1) {
        gcds[idx] = gcd(numbers[idx], numbers[idx + 1]);
    }
}

extern "C" void runFindGCDs(int *gcds, int *numbers, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size - 1 + threadsPerBlock - 1) / threadsPerBlock;
    findGCDs<<<blocksPerGrid, threadsPerBlock>>>(gcds, numbers, size);
}

__device__ bool isPerfectSquare(int num) {
    int root = static_cast<int>(sqrt(num));
    return root * root == num;
}

__global__ void checkPerfectSquares(bool *results, int *nums, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        results[idx] = isPerfectSquare(nums[idx]);
    }
}

extern "C" void runCheckPerfectSquares(bool *results, int *nums, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    checkPerfectSquares<<<blocksPerGrid, threadsPerBlock>>>(results, nums, size);
}

__device__ int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

__global__ void findLCMs(int *lcmResults, int *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size - 1) {
        lcmResults[idx] = lcm(numbers[idx], numbers[idx + 1]);
    }
}

extern "C" void runFindLCMs(int *lcmResults, int *numbers, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size - 1 + threadsPerBlock - 1) / threadsPerBlock;
    findLCMs<<<blocksPerGrid, threadsPerBlock>>>(lcmResults, numbers, size);
}

__device__ int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void calculateFibonacci(int *fibResults, int *indices, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        fibResults[idx] = fibonacci(indices[idx]);
    }
}

extern "C" void runCalculateFibonacci(int *fibResults, int *indices, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    calculateFibonacci<<<blocksPerGrid, threadsPerBlock>>>(fibResults, indices, size);
}

__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

__global__ void calculateFactorials(int *factResults, int *numbers, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        factResults[idx] = factorial(numbers[idx]);
    }
}

extern "C" void runCalculateFactorials(int *factResults, int *numbers, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    calculateFactorials<<<blocksPerGrid, threadsPerBlock>>>(factResults, numbers, size);
}

__device__ bool isArmstrong(int num) {
    int originalNum = num;
    int sum = 0, digits = 0, temp;

    while (temp != 0) {
        temp = num % 10;
        digits++;
        num /= 10;
    }

    num = originalNum;
    while (temp != 0) {
        temp = num % 10;
        sum += pow(temp, digits);
        num /= 10;
    }

    return originalNum == sum;
}

__global__ void checkArmstrongNumbers(bool *results, int *nums, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        results[idx] = isArmstrong(nums[idx]);
    }
}

extern "C" void runCheckArmstrongNumbers(bool *results, int *nums, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    checkArmstrongNumbers<<<blocksPerGrid, threadsPerBlock>>>(results, nums, size);
}

__device__ bool isPalindrome(int num) {
    int originalNum = num;
    int reversedNum = 0;

    while (num != 0) {
        int digit = num % 10;
        reversedNum = reversedNum * 10 + digit;
        num /= 10;
    }

    return originalNum == reversedNum;
}

__global__ void checkPalindromes(bool *results, int *nums, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        results[idx] = isPalindrome(nums[idx]);
    }
}

extern "C" void runCheckPalindromes(bool *results, int *nums, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    checkPalindromes<<<blocksPerGrid, threadsPerBlock>>>(results, nums, size);
}

__device__ bool isTriangular(int num) {
    if (num < 0) return false;

    int n = static_cast<int>(sqrt(2 * num));
    return n * (n + 1) / 2 == num;
}

__global__ void checkTriangularNumbers(bool *results, int *nums, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        results[idx] = isTriangular(nums[idx]);
    }
}

extern "C" void runCheckTriangularNumbers(bool *results, int *nums, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    checkTriangularNumbers<<<blocksPerGrid, threadsPerBlock>>>(results, nums, size);
}
