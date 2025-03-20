#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void generatePrimes(int* primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool isPrime = true;
        for (int i = 2; i <= sqrt(idx); ++i) {
            if (idx % i == 0) {
                isPrime = false;
                break;
            }
        }
        primes[idx] = isPrime ? idx : 0;
    }
}

__global__ void gaussianKernel(float* data, float* output, int width, int height, float sigma) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float dx = nx - x;
                    float dy = ny - y;
                    sum += data[ny * width + nx] * expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
                }
            }
        }
        output[y * width + x] = sum;
    }
}

__global__ void gradientDescent(float* weights, float* targets, float* predictions, int n, float learningRate) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float prediction = weights[idx];
        float error = targets[idx] - prediction;
        weights[idx] += learningRate * error;
    }
}

__global__ void primeFilter(int* numbers, int* primes, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        numbers[idx] = primes[idx] > 0 ? numbers[idx] : 0;
    }
}

__global__ void matrixMultiply(float* A, float* B, float* C, int widthA, int heightA, int widthB) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

__global__ void reluKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void sigmoidKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanhKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void softmaxKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float maxVal = -1e30f;
        for (int i = 0; i < n; ++i) {
            maxVal = fmax(maxVal, input[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            output[i] = expf(input[i] - maxVal);
            sum += output[i];
        }
        for (int i = 0; i < n; ++i) {
            output[i] /= sum;
        }
    }
}

__global__ void addKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void subtractKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void multiplyKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void divideKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void powKernel(float* base, float* exponent, float* result, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = powf(base[idx], exponent[idx]);
    }
}

__global__ void sqrtKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

__global__ void absKernel(float* input, float* output, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabsf(input[idx]);
    }
}

__global__ void maxKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmax(a[idx], b[idx]);
    }
}

__global__ void minKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmin(a[idx], b[idx]);
    }
}

__global__ void dotProductKernel(float* a, float* b, float* result, int n) {
    extern __shared__ float shared[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < n) {
        shared[tid] = a[idx] * b[idx];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

__global__ void crossProductKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 2) {
        c[idx] = a[idx + 1] * b[idx + 2] - a[idx + 2] * b[idx + 1];
        c[idx + 1] = a[idx + 2] * b[idx] - a[idx] * b[idx + 2];
        c[idx + 2] = a[idx] * b[idx + 1] - a[idx + 1] * b[idx];
    }
}

__global__ void normalizeKernel(float* input, float* output, int n) {
    extern __shared__ float shared[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    if (idx < n) {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float norm = sqrtf(shared[0]);
        for (unsigned int i = tid; i < n; i += blockDim.x) {
            output[i] = input[i] / norm;
        }
    }
}

__global__ void elementwiseMultiplyKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwiseDivideKernel(float* a, float* b, float* c, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void transposeKernel(float* input, float* output, int rows, int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

__global__ void transposeKernelShared(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        shared[threadIdx.y * blockDim.x + threadIdx.x] = input[idy * cols + idx];
    }
    __syncthreads();

    if (idx < rows && idy < cols) {
        output[idx * cols + idy] = shared[threadIdx.x * blockDim.y + threadIdx.y];
    }
}

int main() {
    // Example usage
    int n = 1024;
    float* a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));

    // Initialize a and b with some values
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    cudaMemcpy(a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Example kernel launch
    addKernel<<<(n + 255) / 256, 256>>>(a, b, c, n);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
