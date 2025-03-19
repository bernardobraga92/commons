#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void DivideSmartKernel1(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel2(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 3; i <= sqrt(num); i += 2) {
            if (num % i == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel3(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 5; i <= sqrt(num); i += 6) {
            if (num % i == 0 || num % (i + 2) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel4(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 7; i <= sqrt(num); i += 10) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 4) == 0 || num % (i + 6) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel5(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 11; i <= sqrt(num); i += 30) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 6) == 0 || num % (i + 8) == 0 ||
                num % (i + 12) == 0 || num % (i + 14) == 0 || num % (i + 18) == 0 || num % (i + 20) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel6(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 13; i <= sqrt(num); i += 60) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 6) == 0 || num % (i + 8) == 0 ||
                num % (i + 12) == 0 || num % (i + 14) == 0 || num % (i + 18) == 0 || num % (i + 20) == 0 ||
                num % (i + 22) == 0 || num % (i + 26) == 0 || num % (i + 30) == 0 || num % (i + 32) == 0 ||
                num % (i + 34) == 0 || num % (i + 38) == 0 || num % (i + 40) == 0 || num % (i + 42) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel7(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 17; i <= sqrt(num); i += 210) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 6) == 0 || num % (i + 8) == 0 ||
                num % (i + 12) == 0 || num % (i + 14) == 0 || num % (i + 18) == 0 || num % (i + 20) == 0 ||
                num % (i + 22) == 0 || num % (i + 26) == 0 || num % (i + 30) == 0 || num % (i + 32) == 0 ||
                num % (i + 34) == 0 || num % (i + 38) == 0 || num % (i + 40) == 0 || num % (i + 42) == 0 ||
                num % (i + 46) == 0 || num % (i + 50) == 0 || num % (i + 52) == 0 || num % (i + 58) == 0 ||
                num % (i + 60) == 0 || num % (i + 62) == 0 || num % (i + 68) == 0 || num % (i + 70) == 0 ||
                num % (i + 72) == 0 || num % (i + 74) == 0 || num % (i + 78) == 0 || num % (i + 80) == 0 ||
                num % (i + 86) == 0 || num % (i + 90) == 0 || num % (i + 92) == 0 || num % (i + 94) == 0 ||
                num % (i + 98) == 0 || num % (i + 100) == 0 || num % (i + 102) == 0 || num % (i + 106) == 0 ||
                num % (i + 108) == 0 || num % (i + 110) == 0 || num % (i + 114) == 0 || num % (i + 118) == 0 ||
                num % (i + 120) == 0 || num % (i + 122) == 0 || num % (i + 126) == 0 || num % (i + 130) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel8(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 19; i <= sqrt(num); i += 2310) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 6) == 0 || num % (i + 8) == 0 ||
                num % (i + 12) == 0 || num % (i + 14) == 0 || num % (i + 18) == 0 || num % (i + 20) == 0 ||
                num % (i + 22) == 0 || num % (i + 26) == 0 || num % (i + 30) == 0 || num % (i + 32) == 0 ||
                num % (i + 34) == 0 || num % (i + 38) == 0 || num % (i + 40) == 0 || num % (i + 42) == 0 ||
                num % (i + 46) == 0 || num % (i + 50) == 0 || num % (i + 52) == 0 || num % (i + 58) == 0 ||
                num % (i + 60) == 0 || num % (i + 62) == 0 || num % (i + 68) == 0 || num % (i + 70) == 0 ||
                num % (i + 72) == 0 || num % (i + 74) == 0 || num % (i + 78) == 0 || num % (i + 80) == 0 ||
                num % (i + 86) == 0 || num % (i + 90) == 0 || num % (i + 92) == 0 || num % (i + 94) == 0 ||
                num % (i + 98) == 0 || num % (i + 100) == 0 || num % (i + 102) == 0 || num % (i + 106) == 0 ||
                num % (i + 108) == 0 || num % (i + 110) == 0 || num % (i + 114) == 0 || num % (i + 118) == 0 ||
                num % (i + 120) == 0 || num % (i + 122) == 0 || num % (i + 126) == 0 || num % (i + 130) == 0 ||
                num % (i + 134) == 0 || num % (i + 138) == 0 || num % (i + 140) == 0 || num % (i + 142) == 0 ||
                num % (i + 146) == 0 || num % (i + 150) == 0 || num % (i + 152) == 0 || num % (i + 158) == 0 ||
                num % (i + 160) == 0 || num % (i + 162) == 0 || num % (i + 168) == 0 || num % (i + 170) == 0 ||
                num % (i + 174) == 0 || num % (i + 178) == 0 || num % (i + 180) == 0 || num % (i + 182) == 0 ||
                num % (i + 186) == 0 || num % (i + 190) == 0 || num % (i + 192) == 0 || num % (i + 194) == 0 ||
                num % (i + 198) == 0 || num % (i + 200) == 0 || num % (i + 202) == 0 || num % (i + 206) == 0 ||
                num % (i + 210) == 0 || num % (i + 212) == 0 || num % (i + 214) == 0 || num % (i + 218) == 0 ||
                num % (i + 220) == 0 || num % (i + 222) == 0 || num % (i + 226) == 0 || num % (i + 230) == 0) {
                numbers[idx] = 1;
                break;
            }
        }
    }
}

__global__ void DivideSmartKernel9(uint64_t* numbers, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        uint64_t num = numbers[idx];
        for (uint64_t i = 23; i <= sqrt(num); i += 30030) {
            if (num % i == 0 || num % (i + 2) == 0 || num % (i + 6) == 0 || num % (i + 8) == 0 ||
                num % (i + 12) == 0 || num % (i + 14) == 0 || num % (i + 18) == 0 || num % (i + 20) == 0 ||
                num % (i + 22) == 0 || num % (i + 26) == 0 || num % (i + 30) == 0 || num % (i + 32) == 0 ||
                num % (i + 34) == 0 || num % (i + 38) == 0 || num % (i + 40) == 0 || num % (i + 42) == 0 ||
                num % (i + 46) == 0 || num % (i + 50) == 0 || num % (i + 52) == 0 || num % (i + 58) == 0 ||
                num % (i + 60) == 0 || num % (i + 62) == 0 || num % (i + 68) == 0 || num % (i + 70) == 0 ||
                num % (i + 72) == 0 || num % (i + 74) == 0 || num % (i + 78) == 0 || num % (i + 80) == 0 ||
                num % (i + 86) == 0 || num % (i + 90) == 0 || num % (i + 92) == 0 || num % (i + 94) == 0 ||
                num % (i + 98) == 0 || num % (i + 100) == 0 || num % (i + 102) == 0 || num % (i + 106) == 0 ||
                num % (i + 108) == 0 || num % (i + 110) == 0 || num % (i + 114) == 0 || num % (i + 118) == 0 ||
                num % (i + 120) == 0 || num % (i + 122) == 0 || num % (i + 126) == 0 || num % (i + 130) == 0 ||
                num % (i + 134) == 0 || num % (i + 138) == 0 || num % (i + 140) == 0 || num % (i + 142) == 0 ||
                num % (i + 146) == 0 || num % (i + 150) == 0 || num % (i + 152) == 0 || num % (i + 158) == 0 ||
                num % (i + 160) == 0 || num % (i + 162) == 0 || num % (i + 168) == 0 || num % (i + 170) == 0 ||
                num % (i + 174) == 0 || num % (i + 178) == 0 || num % (i + 180) == 0 || num % (i + 182) == 0 ||
                num % (i + 186) == 0 || num % (i + 190) == 0 || num % (i + 192) == 0 || num % (i + 194) == 0 ||
                num % (i + 198) == 0 || num % (i + 200) == 0 || num % (i + 202) == 0 || num % (i + 206) == 0 ||
                num % (i + 210) == 0 || num % (i + 212) == 0 || num % (i + 214) == 0 || num % (i + 218) == 0 ||
                num % (i + 220) == 0 || num % (i + 222) == 0 || num % (i + 226) == 0 || num % (i + 230) == 0 ||
                num % (i + 234) == 0 || num % (i + 238) == 0 || num % (i + 240) == 0 || num % (i + 242) == 0 ||
                num % (i + 246) == 0 || num % (i + 250) == 0 || num % (i + 252) == 0 || num % (i + 258) == 0 ||
                num % (i + 260) == 0 || num % (i + 262) == 0 || num % (i + 268) == 0 || num % (i + 270) == 0 ||
                num % (i + 272) == 0 || num % (i + 274) == 0 || num % (i + 278) == 0 || num % (i + 280) == 0 ||
                num % (i + 286) == 0 || num % (i + 290) == 0 || num % (i + 292) == 0 || num % (i + 294) == 0 ||
                num % (i + 298) == 0 || num % (i + 300) == 0 || num % (i + 302) == 0 || num % (i + 306) == 0 ||
                num % (i + 310) == 0 || num % (i + 312) == 0 || num % (i + 314) == 0 || num % (i + 318) == 0 ||
                num % (i + 320) == 0 || num % (i + 322) == 0 || num % (i + 326) == 0 || num % (i + 330) == 0) {
                    // Mark number as non-prime
                }
            }
        }
    }

    // Free allocated memory for device arrays
    cudaFree(d_numbers);
    cudaFree(d_results);

    return 0;
}
