#include <iostream>
#include <cmath>

__global__ void negGrowthKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = -abs(data[idx]);
    }
}

__global__ void discConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

__global__ void findLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] = -1; // Mark large primes with a negative value
    }
}

__global__ void grothendieckDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void negateLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 0 && isPrime(data[idx])) {
        data[idx] *= -1;
    }
}

__global__ void normalizeDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void shiftLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] += 100; // Shift large primes by a constant
    }
}

__global__ void invertDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = 1 / (a[idx] * b[idx]);
    }
}

__global__ void squareLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] *= data[idx];
    }
}

__global__ void multiplyDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void addLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] += 50; // Add a constant to large primes
    }
}

__global__ void divideDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void subtractLargePrimesKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && isPrime(data[idx])) {
        data[idx] -= 25; // Subtract a constant from large primes
    }
}

__global__ void modDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && b[idx] != 0) {
        c[idx] = a[idx] % b[idx];
    }
}

__global__ void absDiscConvKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = abs(a[idx] - b[idx]);
    }
}

int main() {
    // Example usage
    const int size = 1024;
    int *h_data, *d_data;
    h_data = new int[size];
    cudaMalloc(&d_data, size * sizeof(int));

    for (int i = 0; i < size; ++i) {
        h_data[i] = rand() % 1000 + 1;
    }

    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    negGrowthKernel<<<(size + 255) / 256, 256>>>(d_data, size);
    findLargePrimesKernel<<<(size + 255) / 256, 256>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_data;
    cudaFree(d_data);

    return 0;
}
