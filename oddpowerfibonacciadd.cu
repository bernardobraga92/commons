#include <stdio.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 256

__global__ void isPrimeKernel(int *d_is_prime, int num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        d_is_prime[0] = (num <= 1) ? 0 : 1;
        for (int i = 2; i <= sqrt(num); ++i) {
            if (num % i == 0) {
                d_is_prime[0] = 0;
                break;
            }
        }
    }
}

__global__ void generateOddFibonacciKernel(int *d_fib, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (idx == 0) d_fib[0] = 1;
        else if (idx == 1) d_fib[1] = 1;
        else d_fib[idx] = d_fib[idx - 1] + d_fib[idx - 2];
    }
}

__global__ void addOddFibonacciPrimesKernel(int *d_odd_fib, int *d_is_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (d_odd_fib[idx] % 2 != 0) {
            d_is_prime[0] += d_odd_fib[idx];
        }
    }
}

__global__ void multiplyOddFibonacciPrimesKernel(int *d_odd_fib, int *d_is_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (d_odd_fib[idx] % 2 != 0) {
            d_is_prime[0] *= d_odd_fib[idx];
        }
    }
}

__global__ void checkOddFibonacciPrimeKernel(int *d_odd_fib, int *d_is_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) {
                d_is_prime[0] = 0;
                break;
            }
        }
    }
}

__global__ void countOddFibonacciPrimesKernel(int *d_odd_fib, int *d_count, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicAdd(d_count, 1);
    }
}

__global__ void sumOddFibonacciKernel(int *d_odd_fib, int *d_sum, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicAdd(d_sum, d_odd_fib[idx]);
    }
}

__global__ void maxOddFibonacciKernel(int *d_odd_fib, int *d_max, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicMax(d_max, d_odd_fib[idx]);
    }
}

__global__ void minOddFibonacciKernel(int *d_odd_fib, int *d_min, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicMin(d_min, d_odd_fib[idx]);
    }
}

__global__ void powerOddFibonacciKernel(int *d_odd_fib, int *d_power, int n, int exponent) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        d_power[0] *= pow(d_odd_fib[idx], exponent);
    }
}

__global__ void findLargestOddFibonacciPrimeKernel(int *d_odd_fib, int *d_largest_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicMax(d_largest_prime, temp);
    }
}

__global__ void sumOddFibonacciPrimesKernel(int *d_odd_fib, int *d_is_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicAdd(d_is_prime, temp);
    }
}

__global__ void countOddFibonacciKernel(int *d_odd_fib, int *d_count, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicAdd(d_count, 1);
    }
}

__global__ void multiplyOddFibonacciKernel(int *d_odd_fib, int *d_product, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        atomicMul(d_product, d_odd_fib[idx]);
    }
}

__global__ void findSmallestOddFibonacciPrimeKernel(int *d_odd_fib, int *d_smallest_prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicMin(d_smallest_prime, temp);
    }
}

__global__ void sumSquaresOddFibonacciPrimesKernel(int *d_odd_fib, int *d_sum_squares, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicAdd(d_sum_squares, temp * temp);
    }
}

__global__ void productOddFibonacciPrimesKernel(int *d_odd_fib, int *d_product_primes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicMul(d_product_primes, temp);
    }
}

__global__ void findOddFibonacciPrimeAboveKernel(int *d_odd_fib, int *d_prime_above, int target, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0 && d_odd_fib[idx] > target) {
        for (int i = 2; i <= sqrt(d_odd_fib[idx]); ++i) {
            if (d_odd_fib[idx] % i == 0) break;
        }
        atomicMin(d_prime_above, d_odd_fib[idx]);
    }
}

__global__ void findOddFibonacciPrimeBelowKernel(int *d_odd_fib, int *d_prime_below, int target, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0 && d_odd_fib[idx] < target) {
        for (int i = 2; i <= sqrt(d_odd_fib[idx]); ++i) {
            if (d_odd_fib[idx] % i == 0) break;
        }
        atomicMax(d_prime_below, d_odd_fib[idx]);
    }
}

__global__ void sumCubesOddFibonacciPrimesKernel(int *d_odd_fib, int *d_sum_cubes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicAdd(d_sum_cubes, temp * temp * temp);
    }
}

__global__ void productSquaresOddFibonacciPrimesKernel(int *d_odd_fib, int *d_product_squares, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicMul(d_product_squares, temp * temp);
    }
}

__global__ void countOddFibonacciPrimesKernel(int *d_odd_fib, int *d_count_primes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_odd_fib[idx] % 2 != 0) {
        int temp = d_odd_fib[idx];
        for (int i = 2; i <= sqrt(temp); ++i) {
            if (temp % i == 0) break;
        }
        atomicAdd(d_count_primes, 1);
    }
}
