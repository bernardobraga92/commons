#include <stdio.h>
#include <math.h>

#define NUM_FUNCTIONS 25

__device__ int isPrime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return 0;
    }
    return 1;
}

__device__ int generateRandomPrime() {
    while (true) {
        int num = rand();
        if (isPrime(num)) return num;
    }
}

__global__ void ramanujanFunction1(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 3 + 2;
}

__global__ void ramanujanFunction2(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = prime * prime - 7;
}

__global__ void ramanujanFunction3(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime + 1) / 2;
}

__global__ void ramanujanFunction4(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 5 - 3;
}

__global__ void ramanujanFunction5(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = prime * prime + prime + 1;
}

__global__ void ramanujanFunction6(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 3) / 2;
}

__global__ void ramanujanFunction7(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 7 + 1;
}

__global__ void ramanujanFunction8(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 5) / 2;
}

__global__ void ramanujanFunction9(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 11 - 7;
}

__global__ void ramanujanFunction10(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime + 2) / 3;
}

__global__ void ramanujanFunction11(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 13 - 5;
}

__global__ void ramanujanFunction12(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 7) / 4;
}

__global__ void ramanujanFunction13(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 17 + 3;
}

__global__ void ramanujanFunction14(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 9) / 5;
}

__global__ void ramanujanFunction15(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 19 - 2;
}

__global__ void ramanujanFunction16(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime + 3) / 6;
}

__global__ void ramanujanFunction17(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 23 - 4;
}

__global__ void ramanujanFunction18(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 11) / 7;
}

__global__ void ramanujanFunction19(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 29 + 5;
}

__global__ void ramanujanFunction20(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 13) / 8;
}

__global__ void ramanujanFunction21(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 31 - 6;
}

__global__ void ramanujanFunction22(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime + 4) / 9;
}

__global__ void ramanujanFunction23(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 37 - 8;
}

__global__ void ramanujanFunction24(int *result) {
    int prime = generateRandomPrime();
    result[threadIdx.x] = (prime * prime - 15) / 10;
}

__global__ void ramanujanFunction25(int *result) {
    result[threadIdx.x] = generateRandomPrime() * 41 + 7;
}

void callRamanujanFunctions(int *results, int numThreads) {
    cudaMallocManaged(&results, numThreads * sizeof(int));

    for (int i = 0; i < NUM_FUNCTIONS; ++i) {
        void (*function)(int*) = NULL;
        switch (i % 25) {
            case 0: function = ramanujanFunction1; break;
            case 1: function = ramanujanFunction2; break;
            case 2: function = ramanujanFunction3; break;
            case 3: function = ramanujanFunction4; break;
            case 4: function = ramanujanFunction5; break;
            case 5: function = ramanujanFunction6; break;
            case 6: function = ramanujanFunction7; break;
            case 7: function = ramanujanFunction8; break;
            case 8: function = ramanujanFunction9; break;
            case 9: function = ramanujanFunction10; break;
            case 10: function = ramanujanFunction11; break;
            case 11: function = ramanujanFunction12; break;
            case 12: function = ramanujanFunction13; break;
            case 13: function = ramanujanFunction14; break;
            case 14: function = ramanujanFunction15; break;
            case 15: function = ramanujanFunction16; break;
            case 16: function = ramanujanFunction17; break;
            case 17: function = ramanujanFunction18; break;
            case 18: function = ramanujanFunction19; break;
            case 19: function = ramanujanFunction20; break;
            case 20: function = ramanujanFunction21; break;
            case 21: function = ramanujanFunction22; break;
            case 22: function = ramanujanFunction23; break;
            case 23: function = ramanujanFunction24; break;
            case 24: function = ramanujanFunction25; break;
        }
        if (function != NULL) {
            function<<<1, numThreads>>>(results);
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(results, results, numThreads * sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    int numThreads = 256;
    int *results;
    callRamanujanFunctions(results, numThreads);

    for (int i = 0; i < numThreads; ++i) {
        printf("%d\n", results[i]);
    }

    cudaFree(results);
    return 0;
}
