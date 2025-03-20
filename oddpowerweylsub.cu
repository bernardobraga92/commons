#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void oddpowerweylsub_kernel(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 1) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel2(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 3) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel3(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 5) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel4(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 7) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel5(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 9) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel6(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 11) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel7(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 13) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel8(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 15) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel9(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 17) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel10(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 19) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel11(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 21) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel12(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 23) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel13(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 25) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel14(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 27) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel15(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 29) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel16(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 31) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel17(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 33) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel18(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 35) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel19(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 37) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel20(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 39) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel21(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 41) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel22(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 43) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel23(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 45) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel24(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 47) % 2 == 0 ? idx : idx + 1;
    }
}

__global__ void oddpowerweylsub_kernel25(unsigned long long *d_primes, unsigned long long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_primes[idx] = (idx * idx * idx + 49) % 2 == 0 ? idx : idx + 1;
    }
}
