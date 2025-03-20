#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <ctime>

__global__ void NegMellinConvKernel1(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 1) | 1;
    }
}

__global__ void NegMellinConvKernel2(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] = (numbers[idx] * 3 + 7) % 1009;
    }
}

__global__ void NegMellinConvKernel3(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 5);
    }
}

__global__ void NegMellinConvKernel4(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 2) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel5(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 3);
    }
}

__global__ void NegMellinConvKernel6(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 1) | 1;
    }
}

__global__ void NegMellinConvKernel7(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 4);
    }
}

__global__ void NegMellinConvKernel8(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 3) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel9(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 5);
    }
}

__global__ void NegMellinConvKernel10(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 2) | 1;
    }
}

__global__ void NegMellinConvKernel11(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 6);
    }
}

__global__ void NegMellinConvKernel12(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 4) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel13(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 7);
    }
}

__global__ void NegMellinConvKernel14(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 5) | 1;
    }
}

__global__ void NegMellinConvKernel15(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 8);
    }
}

__global__ void NegMellinConvKernel16(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 6) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel17(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 9);
    }
}

__global__ void NegMellinConvKernel18(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 7) | 1;
    }
}

__global__ void NegMellinConvKernel19(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 10);
    }
}

__global__ void NegMellinConvKernel20(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 8) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel21(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 11);
    }
}

__global__ void NegMellinConvKernel22(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 9) | 1;
    }
}

__global__ void NegMellinConvKernel23(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 12);
    }
}

__global__ void NegMellinConvKernel24(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (numbers[idx] << 10) ^ 0x7FF;
    }
}

__global__ void NegMellinConvKernel25(int* numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= (numbers[idx] >> 13);
    }
}
