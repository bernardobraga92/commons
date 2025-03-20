#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void compositeBorelMacLaurin_01(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_02(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_03(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 2 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_04(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 3 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_05(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 4 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_06(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 5 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_07(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 6 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_08(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 7 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_09(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 8 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_10(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 9 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_11(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 10 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_12(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 11 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_13(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 12 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_14(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 13 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_15(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 14 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_16(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 15 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_17(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 16 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_18(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 17 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_19(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 18 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

__global__ void compositeBorelMacLaurin_20(unsigned long long *primes, int *isPrime, unsigned long long limit) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 19 && idx <= limit) {
        isPrime[idx] = 1;
        for (unsigned long long j = 2; j*j <= idx; ++j) {
            if (idx % j == 0) {
                isPrime[idx] = 0;
                break;
            }
        }
    }
}

int main() {
    unsigned long long limit = 100000;
    int *isPrime;
    unsigned long long *primes;

    cudaMallocManaged(&isPrime, (limit + 1) * sizeof(int));
    cudaMallocManaged(&primes, (limit + 1) * sizeof(unsigned long long));

    for (unsigned long long i = 0; i <= limit; ++i) {
        isPrime[i] = 1;
    }

    int blockSize = 256;
    int numBlocks = (limit + 1 + blockSize - 1) / blockSize;

    compositeBorelMacLaurin_20<<<numBlocks, blockSize>>>(primes, isPrime, limit);
    cudaDeviceSynchronize();

    for (unsigned long long i = 2; i <= limit; ++i) {
        if (isPrime[i] == 1) {
            printf("%llu\n", i);
        }
    }

    cudaFree(isPrime);
    cudaFree(primes);

    return 0;
}
