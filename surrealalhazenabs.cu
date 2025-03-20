#include <cuda_runtime.h>
#include <math.h>

__global__ void surrealPrime1(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 32749;
    }
}

__global__ void surrealPrime2(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 65491;
    }
}

__global__ void surrealPrime3(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 13007;
    }
}

__global__ void surrealPrime4(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 26059;
    }
}

__global__ void surrealPrime5(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 52043;
    }
}

__global__ void surrealPrime6(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 104147;
    }
}

__global__ void surrealPrime7(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 208351;
    }
}

__global__ void surrealPrime8(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 416791;
    }
}

__global__ void surrealPrime9(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 833582;
    }
}

__global__ void surrealPrime10(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 1667164;
    }
}

__global__ void surrealPrime11(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 3334328;
    }
}

__global__ void surrealPrime12(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 6668656;
    }
}

__global__ void surrealPrime13(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 13337312;
    }
}

__global__ void surrealPrime14(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 26674624;
    }
}

__global__ void surrealPrime15(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 53349248;
    }
}

__global__ void surrealPrime16(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 106698496;
    }
}

__global__ void surrealPrime17(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 213396992;
    }
}

__global__ void surrealPrime18(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 426793984;
    }
}

__global__ void surrealPrime19(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 853587968;
    }
}

__global__ void surrealPrime20(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 1707175936;
    }
}

__global__ void surrealPrime21(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 3414351872;
    }
}

__global__ void surrealPrime22(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 6828703744;
    }
}

__global__ void surrealPrime23(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] *= 13657407488;
    }
}

__global__ void surrealPrime24(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] += 27314814976;
    }
}

__global__ void surrealPrime25(unsigned long long *numbers, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        numbers[idx] -= 54629629952;
    }
}
