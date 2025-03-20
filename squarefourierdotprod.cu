#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__device__ bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void findPrimes(int* numbers, int* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        if (isPrime(numbers[idx])) {
            results[idx] = 1;
        } else {
            results[idx] = 0;
        }
    }
}

__global__ void squareNumbers(int* numbers, int* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = numbers[idx] * numbers[idx];
    }
}

__global__ void dotProduct(float* a, float* b, float* result, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < N) {
        sum += a[idx] * b[idx];
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

__global__ void fftKernel(float2* input, float2* output, int N, int logN) {
    extern __shared__ float2 shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int r = __ffs(tid ^ (tid >> 1));
        int p = (r > logN) ? 0 : (1 << (logN - r));

        shared[tid] = input[idx];
        __syncthreads();

        for (int s = 2; s <= N; s <<= 1) {
            if ((tid & (s - 1)) == 0) {
                int m = s >> 1;
                float2 t = make_float2(__sinf(tid * 2.0f * M_PI / N), __cosf(tid * 2.0f * M_PI / N));
                shared[tid] += complexMul(shared[tid + m], t);
            }
            __syncthreads();
        }

        output[idx] = shared[tid];
    }
}

__global__ void complexMulKernel(float2 a, float2 b, float2* result) {
    int tid = threadIdx.x;
    if (tid == 0) {
        result[0] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
}

__global__ void generateRandomNumbers(int* numbers, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        numbers[idx] = rand() % 1000000; // Generate random numbers
    }
}

__global__ void addKernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void subtractKernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void multiplyKernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void divideKernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && b[idx] != 0) {
        c[idx] = a[idx] / b[idx];
    } else {
        c[idx] = 0.0f;
    }
}

__global__ void maxKernel(float* a, float* result, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        shared[tid] = a[idx];
    } else {
        shared[tid] = -FLT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, __float_as_int(shared[0]));
    }
}

__global__ void minKernel(float* a, float* result, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        shared[tid] = a[idx];
    } else {
        shared[tid] = FLT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            shared[tid] = min(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(result, __float_as_int(shared[0]));
    }
}

__global__ void normalizeKernel(float* a, float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && b[idx] != 0) {
        a[idx] /= b[idx];
    }
}

__global__ void transposeKernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = input[x * height + y];
    }
}

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int widthA, int heightA, int widthB, int heightB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

__global__ void gaussianBlurKernel(float* input, float* output, int width, int height, float sigma) {
    extern __shared__ float shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx] * expf(-(i * i + j * j) / (2.0f * sigma * sigma));
                }
            }
        }
        output[y * width + x] = sum;
    }
}

__global__ void edgeDetectionKernel(float* input, float* output, int width, int height) {
    extern __shared__ float shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx] * (-1);
                }
            }
        }
        output[y * width + x] = abs(sum);
    }
}

__global__ void histogramEqualizationKernel(unsigned char* input, unsigned char* output, int width, int height, float* hist) {
    extern __shared__ float shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelValue = input[y * width + x];
        output[y * width + x] = static_cast<unsigned char>(255.0f * hist[pixelValue]);
    }
}

__global__ void cannyEdgeDetectionKernel(float* input, float* output, int width, int height) {
    extern __shared__ float shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx] * (-1);
                }
            }
        }
        output[y * width + x] = abs(sum);
    }
}

__global__ void houghTransformKernel(float* input, float* output, int width, int height) {
    extern __shared__ float shared[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx] * (-1);
                }
            }
        }
        output[y * width + x] = abs(sum);
    }
}

int main() {
    // Example usage of the kernels
    return 0;
}
