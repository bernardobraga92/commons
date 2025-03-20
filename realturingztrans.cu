#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_THREADS 256

__global__ void realturingztrans_1(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += idx * idx;
    }
}

__global__ void realturingztrans_2(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 17 + idx % 5;
    }
}

__global__ void realturingztrans_3(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx << (idx % 8);
    }
}

__global__ void realturingztrans_4(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (1ULL << 32) - idx;
    }
}

__global__ void realturingztrans_5(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= (idx * idx) % 7;
    }
}

__global__ void realturingztrans_6(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= (1ULL << 32) + idx;
    }
}

__global__ void realturingztrans_7(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx >> (idx % 16);
    }
}

__global__ void realturingztrans_8(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += idx * 31;
    }
}

__global__ void realturingztrans_9(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= (idx * idx) % 5 + 7;
    }
}

__global__ void realturingztrans_10(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx << (idx % 32);
    }
}

__global__ void realturingztrans_11(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (1ULL << 64) - idx;
    }
}

__global__ void realturingztrans_12(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= idx * 37 % 11 + 5;
    }
}

__global__ void realturingztrans_13(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx >> (idx % 64);
    }
}

__global__ void realturingztrans_14(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += idx * 41;
    }
}

__global__ void realturingztrans_15(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= (idx * idx) % 7 + 13;
    }
}

__global__ void realturingztrans_16(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx << (idx % 8);
    }
}

__global__ void realturingztrans_17(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += idx * 43;
    }
}

__global__ void realturingztrans_18(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= idx * 47 % 11 + 19;
    }
}

__global__ void realturingztrans_19(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx >> (idx % 16);
    }
}

__global__ void realturingztrans_20(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += idx * 53;
    }
}

__global__ void realturingztrans_21(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= (idx * idx) % 5 + 29;
    }
}

__global__ void realturingztrans_22(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx << (idx % 32);
    }
}

__global__ void realturingztrans_23(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += (1ULL << 64) - idx;
    }
}

__global__ void realturingztrans_24(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= idx * 59 % 11 + 37;
    }
}

__global__ void realturingztrans_25(unsigned long long *numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] ^= idx >> (idx % 64);
    }
}
