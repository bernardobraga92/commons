#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void boundedstieltjesjacobian_kernel(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel2(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 3; i <= sqrt(idx); i += 2) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 2)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel3(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i < sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel4(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= idx / 2; ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel5(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i < idx; ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel6(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel7(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel8(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel9(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel10(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel11(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel12(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel13(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel14(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel15(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel16(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel17(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel18(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel19(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}

__global__ void boundedstieltjesjacobian_kernel20(int* primes, int limit) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= limit) return;

    bool is_prime = true;
    for (int i = 2; i <= sqrt(idx); ++i) {
        if (idx % i == 0) {
            is_prime = false;
            break;
        }
    }
    if (is_prime && (idx > 1)) primes[idx] = idx;
}
