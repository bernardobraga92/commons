#include <cuda_runtime.h>
#include <cmath>

__global__ void irrationalMarkovFloorDiv1(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 2.718) / log(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv2(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((data[idx] * M_PI) / exp(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv3(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor(exp(log(data[idx]) + 1.618) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv4(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log2(data[idx]) + 1.414) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv5(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 1.732) / log2(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv6(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((exp(data[idx]) + 0.577) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv7(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log1p(data[idx]) + 0.693) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv8(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.307) / exp(log1p(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv9(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((expm1(data[idx]) + 0.718) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv10(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.434) / expm1(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv11(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log10(data[idx]) + 0.301) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv12(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.699) / exp(log10(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv13(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((exp2(data[idx]) + 0.693) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv14(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.301) / exp2(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv15(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log2(data[idx]) + 0.693) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv16(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.301) / log2(exp(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv17(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((exp(log(data[idx]) + 0.693) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv18(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.434) / exp(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv19(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log(data[idx]) + 0.301) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv20(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.693) / exp(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv21(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((exp(log(data[idx]) + 0.301) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv22(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.693) / exp(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv23(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((log(data[idx]) + 0.434) / sqrt(data[idx]));
    }
}

__global__ void irrationalMarkovFloorDiv24(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((sqrt(data[idx]) + 0.301) / exp(log(data[idx])));
    }
}

__global__ void irrationalMarkovFloorDiv25(unsigned long long *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = floor((exp(log(data[idx]) + 0.693) / sqrt(data[idx]));
    }
}
