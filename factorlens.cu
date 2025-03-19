#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS_PER_BLOCK 256

__global__ void FactorLens_InitArray(int* array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) array[idx] = idx + 1;
}

__global__ void FactorLens_IsPrime(int* numbers, bool* is_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int n = numbers[idx];
        if (n <= 1) is_prime[idx] = false;
        else {
            is_prime[idx] = true;
            for (int i = 2; i <= sqrt(n); ++i) {
                if (n % i == 0) {
                    is_prime[idx] = false;
                    break;
                }
            }
        }
    }
}

__global__ void FactorLens_EratosthenesSieve(int* numbers, bool* is_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && idx > 1) {
        for (int i = idx * idx; i <= size; i += idx) {
            if (numbers[i] % idx == 0) {
                is_prime[i] = false;
            }
        }
    }
}

__global__ void FactorLens_CountPrimes(bool* is_prime, int size, int* count) {
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        if (is_prime[tid]) atomicAdd(count, 1);
    }
}

__global__ void FactorLens_GeneratePrimes(int* numbers, bool* is_prime, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && is_prime[idx]) {
        numbers[idx] = idx + 1;
    } else {
        numbers[idx] = 0;
    }
}

__global__ void FactorLens_SumPrimes(int* primes, bool* is_prime, int size, int* sum) {
    __shared__ int shared_sum[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(sum, primes[tid]);
    }
}

__global__ void FactorLens_MaxPrime(int* numbers, bool* is_prime, int size, int* max_prime) {
    __shared__ int shared_max[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicMax(max_prime, numbers[tid]);
    }
}

__global__ void FactorLens_MinPrime(int* numbers, bool* is_prime, int size, int* min_prime) {
    __shared__ int shared_min[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicMin(min_prime, numbers[tid]);
    }
}

__global__ void FactorLens_AvgPrime(int* primes, bool* is_prime, int size, float* avg) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_count[0], 1);
    }
    if (threadIdx.x == 0) {
        if (shared_count[0] > 0) *avg = shared_sum[0] / shared_count[0];
    }
}

__global__ void FactorLens_MedianPrime(int* primes, bool* is_prime, int size, float* median) {
    __shared__ int shared_primes[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        shared_primes[threadIdx.x] = primes[tid];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (shared_primes[threadIdx.x] > shared_primes[threadIdx.x + stride]) {
                int temp = shared_primes[threadIdx.x];
                shared_primes[threadIdx.x] = shared_primes[threadIdx.x + stride];
                shared_primes[threadIdx.x + stride] = temp;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *median = shared_primes[blockDim.x / 2];
    }
}

__global__ void FactorLens_ModePrime(int* primes, bool* is_prime, int size, int* mode) {
    __shared__ int shared_mode[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_count[primes[tid]], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int max_count = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            if (shared_count[i] > max_count) {
                max_count = shared_count[i];
                *mode = i;
            }
        }
    }
}

__global__ void FactorLens_StandardDeviationPrime(int* primes, bool* is_prime, int size, float* std_dev) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        *std_dev = sqrtf((shared_square_sum[0] / shared_count[0]) - powf(mean, 2));
    }
}

__global__ void FactorLens_VariancePrime(int* primes, bool* is_prime, int size, float* variance) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        *variance = (shared_square_sum[0] / shared_count[0]) - powf(mean, 2);
    }
}

__global__ void FactorLens_SkewnessPrime(int* primes, bool* is_prime, int size, float* skewness) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_cube_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_cube_sum[0], powf(primes[tid], 3));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        float variance = (shared_square_sum[0] / shared_count[0]) - powf(mean, 2);
        float std_dev = sqrtf(variance);
        *skewness = ((shared_cube_sum[0] / shared_count[0]) - 3 * mean * variance - powf(mean, 3)) / (powf(std_dev, 3) * shared_count[0]);
    }
}

__global__ void FactorLens_KurtosisPrime(int* primes, bool* is_prime, int size, float* kurtosis) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_cube_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_fourth_power_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_cube_sum[0], powf(primes[tid], 3));
        atomicAdd(&shared_fourth_power_sum[0], powf(primes[tid], 4));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        float variance = (shared_square_sum[0] / shared_count[0]) - powf(mean, 2);
        *kurtosis = ((shared_fourth_power_sum[0] / shared_count[0]) - 4 * mean * shared_cube_sum[0] + 6 * powf(mean, 2) * shared_square_sum[0] - 3 * powf(mean, 4)) / (powf(variance, 2) * shared_count[0]);
    }
}

__global__ void FactorLens_QuantilePrime(int* primes, bool* is_prime, int size, float q, float* quantile) {
    __shared__ int shared_primes[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        shared_primes[threadIdx.x] = primes[tid];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (shared_primes[threadIdx.x] > shared_primes[threadIdx.x + stride]) {
                int temp = shared_primes[threadIdx.x];
                shared_primes[threadIdx.x] = shared_primes[threadIdx.x + stride];
                shared_primes[threadIdx.x + stride] = temp;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *quantile = shared_primes[int(q * (blockDim.x - 1))];
    }
}

__global__ void FactorLens_RangePrime(int* primes, bool* is_prime, int size, int* range) {
    __shared__ int min_val;
    __shared__ int max_val;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicMin(&min_val, primes[tid]);
        atomicMax(&max_val, primes[tid]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *range = max_val - min_val;
    }
}

__global__ void FactorLens_InterquartileRangePrime(int* primes, bool* is_prime, int size, float* iqr) {
    __shared__ int shared_primes[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        shared_primes[threadIdx.x] = primes[tid];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (shared_primes[threadIdx.x] > shared_primes[threadIdx.x + stride]) {
                int temp = shared_primes[threadIdx.x];
                shared_primes[threadIdx.x] = shared_primes[threadIdx.x + stride];
                shared_primes[threadIdx.x + stride] = temp;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float q1 = shared_primes[int(0.25 * (blockDim.x - 1))];
        float q3 = shared_primes[int(0.75 * (blockDim.x - 1))];
        *iqr = q3 - q1;
    }
}

__global__ void FactorLens_CoefficientOfVariationPrime(int* primes, bool* is_prime, int size, float* cv) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        float std_dev = sqrt((shared_square_sum[0] - shared_count[0] * powf(mean, 2)) / (shared_count[0] - 1));
        *cv = (std_dev / mean) * 100;
    }
}

__global__ void FactorLens_SkewnessPrime(int* primes, bool* is_prime, int size, float* skewness) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_cube_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_cube_sum[0], powf(primes[tid], 3));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        float std_dev = sqrt((shared_square_sum[0] - shared_count[0] * powf(mean, 2)) / (shared_count[0] - 1));
        float skewness_val = (shared_cube_sum[0] - 3 * mean * shared_square_sum[0] + 2 * powf(mean, 3) * shared_count[0]) / (powf(std_dev, 3) * shared_count[0]);
        *skewness = skewness_val;
    }
}

__global__ void FactorLens_KurtosisPrime(int* primes, bool* is_prime, int size, float* kurtosis) {
    __shared__ float shared_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_square_sum[MAX_THREADS_PER_BLOCK];
    __shared__ float shared_fourth_power_sum[MAX_THREADS_PER_BLOCK];
    __shared__ int shared_count[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size && is_prime[tid]) {
        atomicAdd(&shared_sum[0], primes[tid]);
        atomicAdd(&shared_square_sum[0], powf(primes[tid], 2));
        atomicAdd(&shared_fourth_power_sum[0], powf(primes[tid], 4));
        atomicAdd(&shared_count[0], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mean = shared_sum[0] / shared_count[0];
        float std_dev = sqrt((shared_square_sum[0] - shared_count[0] * powf(mean, 2)) / (shared_count[0] - 1));
        float kurtosis_val = (shared_fourth_power_sum[0] - 4 * mean * shared_cube_sum[0] + 6 * powf(mean, 2) * shared_square_sum[0] - 3 * powf(mean, 4)) / (powf(std_dev, 4) * shared_count[0]);
        *kurtosis = kurtosis_val;
    }
}

int main() {
    // Initialize the random number generator
    srand(time(NULL));

    // Allocate memory for the array of prime numbers and the is_prime flags
    const int size = 1024;
    int* primes;
    bool* is_prime;
    cudaMalloc(&primes, size * sizeof(int));
    cudaMalloc(&is_prime, size * sizeof(bool));

    // Generate random numbers and check for primality
    generate<<<(size + 255) / 256, 256>>>(primes, is_prime, size);

    // Allocate memory for the results of the statistical functions
    float* mean;
    float* std_dev;
    float* median;
    float* mode;
    float* skewness;
    float* kurtosis;
    cudaMalloc(&mean, sizeof(float));
    cudaMalloc(&std_dev, sizeof(float));
    cudaMalloc(&median, sizeof(float));
    cudaMalloc(&mode, sizeof(float));
    cudaMalloc(&skewness, sizeof(float));
    cudaMalloc(&kurtosis, sizeof(float));

    // Calculate the mean
    calculate_mean<<<(size + 255) / 256, 256>>>(primes, is_prime, size, mean);

    // Calculate the standard deviation
    calculate_std_dev<<<(size + 255) / 256, 256>>>(primes, is_prime, size, std_dev);

    // Calculate the median
    calculate_median<<<(size + 255) / 256, 256>>>(primes, is_prime, size, median);

    // Calculate the mode
    calculate_mode<<<(size + 255) / 256, 256>>>(primes, is_prime, size, mode);

    // Calculate the skewness
    calculate_skewness<<<(size + 255) / 256, 256>>>(primes, is_prime, size, skewness);

    // Calculate the kurtosis
    calculate_kurtosis<<<(size + 255) / 256, 256>>>(primes, is_prime, size, kurtosis);

    // Copy the results from the device to the host
    float h_mean, h_std_dev, h_median, h_mode, h_skewness, h_kurtosis;
    cudaMemcpy(&h_mean, mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_std_dev, std_dev, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_median, median, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mode, mode, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_skewness, skewness, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_kurtosis, kurtosis, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Mean: %f\n", h_mean);
    printf("Standard Deviation: %f\n", h_std_dev);
    printf("Median: %f\n", h_median);
    printf("Mode: %f\n", h_mode);
    printf("Skewness: %f\n", h_skewness);
    printf("Kurtosis: %f\n", h_kurtosis);

    // Free the memory allocated for the arrays
    cudaFree(primes);
    cudaFree(is_prime);
    cudaFree(mean);
    cudaFree(std_dev);
    cudaFree(median);
    cudaFree(mode);
    cudaFree(skewness);
    cudaFree(kurtosis);

    return 0;
}
