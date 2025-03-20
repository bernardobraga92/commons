#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void compositeJacobiVecProj1(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 234567891 + 123456789) % 987654321;
    }
}

__global__ void compositeJacobiVecProj2(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 345678901 + 234567890) % 876543219;
    }
}

__global__ void compositeJacobiVecProj3(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 456789012 + 345678901) % 765432109;
    }
}

__global__ void compositeJacobiVecProj4(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 567890123 + 456789012) % 654321097;
    }
}

__global__ void compositeJacobiVecProj5(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 678901234 + 567890123) % 543210987;
    }
}

__global__ void compositeJacobiVecProj6(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 789012345 + 678901234) % 432109876;
    }
}

__global__ void compositeJacobiVecProj7(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 890123456 + 789012345) % 321098765;
    }
}

__global__ void compositeJacobiVecProj8(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 901234567 + 890123456) % 210987654;
    }
}

__global__ void compositeJacobiVecProj9(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 1234567890 + 901234567) % 109876543;
    }
}

__global__ void compositeJacobiVecProj10(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 2345678901 + 1234567890) % 198765432;
    }
}

__global__ void compositeJacobiVecProj11(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 3456789012 + 2345678901) % 187654321;
    }
}

__global__ void compositeJacobiVecProj12(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 4567890123 + 3456789012) % 176543210;
    }
}

__global__ void compositeJacobiVecProj13(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 5678901234 + 4567890123) % 165432109;
    }
}

__global__ void compositeJacobiVecProj14(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 6789012345 + 5678901234) % 154321098;
    }
}

__global__ void compositeJacobiVecProj15(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 7890123456 + 6789012345) % 143210987;
    }
}

__global__ void compositeJacobiVecProj16(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 8901234567 + 7890123456) % 132109876;
    }
}

__global__ void compositeJacobiVecProj17(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 9012345678 + 8901234567) % 121098765;
    }
}

__global__ void compositeJacobiVecProj18(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 12345678901 + 9012345678) % 110987654;
    }
}

__global__ void compositeJacobiVecProj19(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 23456789012 + 12345678901) % 100987653;
    }
}

__global__ void compositeJacobiVecProj20(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 34567890123 + 23456789012) % 908765432;
    }
}

__global__ void compositeJacobiVecProj21(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 45678901234 + 34567890123) % 807654321;
    }
}

__global__ void compositeJacobiVecProj22(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 56789012345 + 45678901234) % 706543210;
    }
}

__global__ void compositeJacobiVecProj23(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 67890123456 + 56789012345) % 605432109;
    }
}

__global__ void compositeJacobiVecProj24(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 78901234567 + 67890123456) % 504321098;
    }
}

__global__ void compositeJacobiVecProj25(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 89012345678 + 78901234567) % 403210987;
    }
}

__global__ void compositeJacobiVecProj26(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 90123456789 + 89012345678) % 302109876;
    }
}

__global__ void compositeJacobiVecProj27(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 123456789012 + 90123456789) % 201098765;
    }
}

__global__ void compositeJacobiVecProj28(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 234567890123 + 123456789012) % 100987654;
    }
}

__global__ void compositeJacobiVecProj29(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 345678901234 + 234567890123) % 90876543;
    }
}

__global__ void compositeJacobiVecProj30(unsigned long *d_numbers, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_numbers[idx] = (d_numbers[idx] * 456789012345 + 345678901234) % 80765432;
    }
}
