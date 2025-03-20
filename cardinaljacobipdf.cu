#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void cardinaljacobipdf_0(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 1729;
    }
}

__global__ void cardinaljacobipdf_1(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 3.14159f;
    }
}

__global__ void cardinaljacobipdf_2(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 666;
    }
}

__global__ void cardinaljacobipdf_3(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 1337;
    }
}

__global__ void cardinaljacobipdf_4(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 2023;
    }
}

__global__ void cardinaljacobipdf_5(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 2.71828f;
    }
}

__global__ void cardinaljacobipdf_6(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 420;
    }
}

__global__ void cardinaljacobipdf_7(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 987654321;
    }
}

__global__ void cardinaljacobipdf_8(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 10007;
    }
}

__global__ void cardinaljacobipdf_9(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 1.61803f;
    }
}

__global__ void cardinaljacobipdf_10(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 7;
    }
}

__global__ void cardinaljacobipdf_11(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 65535;
    }
}

__global__ void cardinaljacobipdf_12(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 123456789;
    }
}

__global__ void cardinaljacobipdf_13(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 0.57721f;
    }
}

__global__ void cardinaljacobipdf_14(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 3.14;
    }
}

__global__ void cardinaljacobipdf_15(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 2718281828459045235;
    }
}

__global__ void cardinaljacobipdf_16(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 1618033988749895;
    }
}

__global__ void cardinaljacobipdf_17(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 1.41421f;
    }
}

__global__ void cardinaljacobipdf_18(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 665772;
    }
}

__global__ void cardinaljacobipdf_19(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 8675309;
    }
}

__global__ void cardinaljacobipdf_20(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 987654321;
    }
}

__global__ void cardinaljacobipdf_21(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] *= 1.73205f;
    }
}

__global__ void cardinaljacobipdf_22(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] -= 498;
    }
}

__global__ void cardinaljacobipdf_23(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] ^= 1234567890123456789;
    }
}

__global__ void cardinaljacobipdf_24(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += 78901234567890;
    }
}
