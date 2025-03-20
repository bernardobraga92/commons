#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void sparseEuclideanCorrelationKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                data[idx] = -1;
                break;
            }
        }
    }
}

void sparseEuclideanCorrelation(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sparseEuclideanCorrelationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void generateRandomPrimesKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        int candidate = curand(&state) % 10000000 + 2;
        for (int i = 2; i <= sqrt(candidate); ++i) {
            if (candidate % i == 0) {
                candidate = -1;
                break;
            }
        }
        data[idx] = candidate;
    }
}

void generateRandomPrimes(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    generateRandomPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void isPrimeKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] <= 1) {
            data[idx] = 0;
            return;
        }
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                data[idx] = 0;
                break;
            }
        }
    }
}

void isPrime(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    isPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void nextPrimeKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int candidate = data[idx] + 1;
        while (true) {
            bool is_prime = true;
            for (int i = 2; i <= sqrt(candidate); ++i) {
                if (candidate % i == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                data[idx] = candidate;
                break;
            }
            candidate++;
        }
    }
}

void nextPrime(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    nextPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void countPrimesKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                atomicSub(&data[size], 1);
                break;
            }
        }
    } else if (idx < size && data[idx] <= 1) {
        atomicSub(&data[size], 1);
    }
}

void countPrimes(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    countPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_data[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void sumPrimesKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                atomicSub(&data[size], data[idx]);
                break;
            }
        }
    } else if (idx < size && data[idx] <= 1) {
        atomicSub(&data[size], data[idx]);
    }
}

void sumPrimes(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sumPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_data[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void filterPrimesKernel(int* data, int* result, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                return;
            }
        }
        atomicAdd(&result[atomicAdd(&result[size], 1)], data[idx]);
    }
}

void filterPrimes(int* h_data, int size) {
    int* d_data;
    int* d_result;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMalloc((void**)&d_result, (size + 1) * sizeof(int));

    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    filterPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_result[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);
}

__global__ void sortPrimesKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                return;
            }
        }
        int temp;
        for (int j = idx + 1; j < size; ++j) {
            if (data[j] > 1 && data[j] < data[idx]) {
                temp = data[idx];
                data[idx] = data[j];
                data[j] = temp;
            }
        }
    }
}

void sortPrimes(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    sortPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void multiplyPrimesKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                return;
            }
        }
        atomicMul(&data[size], data[idx]);
    }
}

void multiplyPrimes(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiplyPrimesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_data[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void findMaxPrimeKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                return;
            }
        }
        atomicMax(&data[size], data[idx]);
    }
}

void findMaxPrime(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findMaxPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_data[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void findMinPrimeKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                return;
            }
        }
        atomicMin(&data[size], data[idx]);
    }
}

void findMinPrime(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, (size + 1) * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    findMinPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data[size], &d_data[size], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

__global__ void checkPrimeKernel(int* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] > 1) {
        for (int i = 2; i <= sqrt(data[idx]); ++i) {
            if (data[idx] % i == 0) {
                data[idx] = 0;
                return;
            }
        }
    } else {
        data[idx] = 0;
    }
}

void checkPrime(int* h_data, int size) {
    int* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}
