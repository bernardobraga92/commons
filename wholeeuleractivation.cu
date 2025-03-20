#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_THREADS 256

__global__ void wholeeuleractivation1(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

__global__ void wholeeuleractivation2(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 3;
    }
}

__global__ void wholeeuleractivation3(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 5;
    }
}

__global__ void wholeeuleractivation4(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 7;
    }
}

__global__ void wholeeuleractivation5(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 11;
    }
}

__global__ void wholeeuleractivation6(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 13;
    }
}

__global__ void wholeeuleractivation7(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 17;
    }
}

__global__ void wholeeuleractivation8(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 19;
    }
}

__global__ void wholeeuleractivation9(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 23;
    }
}

__global__ void wholeeuleractivation10(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 29;
    }
}

__global__ void wholeeuleractivation11(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 31;
    }
}

__global__ void wholeeuleractivation12(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 37;
    }
}

__global__ void wholeeuleractivation13(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 41;
    }
}

__global__ void wholeeuleractivation14(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 43;
    }
}

__global__ void wholeeuleractivation15(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 47;
    }
}

__global__ void wholeeuleractivation16(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 53;
    }
}

__global__ void wholeeuleractivation17(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 59;
    }
}

__global__ void wholeeuleractivation18(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 61;
    }
}

__global__ void wholeeuleractivation19(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 67;
    }
}

__global__ void wholeeuleractivation20(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 71;
    }
}

__global__ void wholeeuleractivation21(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 73;
    }
}

__global__ void wholeeuleractivation22(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 79;
    }
}

__global__ void wholeeuleractivation23(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 83;
    }
}

__global__ void wholeeuleractivation24(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] -= 89;
    }
}

__global__ void wholeeuleractivation25(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 97;
    }
}
