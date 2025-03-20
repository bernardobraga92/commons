#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#define THREADS_PER_BLOCK 256

__global__ void bernoulliKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        primes[idx] = curand(&state) % 1000000007;
    }
}

__global__ void bezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += idx * idx * idx;
    }
}

__global__ void bernoulliBezCompositeKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = ((curand(&state) % 1000000007) * (idx * idx * idx)) + idx;
    }
}

__global__ void denseBernoulliKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void denseBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += idx * idx * idx * curand(&state);
    }
}

__global__ void bernoulliDensePrimeKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = ((curand(&state) % 1000000007) * (idx * idx)) + curand(&state);
    }
}

__global__ void bezierDensePrimeKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] = ((curand(&state) % 1000000007) * (idx * idx)) + curand(&state);
    }
}

__global__ void bernoulliDenseKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierDenseKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierBernoulliKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliDenseBernKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierDenseBernKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliDenseBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierDenseBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliBernKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierBernKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliBezierBernKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierBernBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bernoulliDenseBernBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

__global__ void bezierDenseBernBezierKernel(unsigned long long *primes, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        primes[idx] += curand(&state) % 1000000007;
    }
}

int main() {
    const int size = 1 << 24; // 16 million elements
    thrust::device_vector<unsigned long long> d_primes(size);
    unsigned long long *primes = thrust::raw_pointer_cast(d_primes.data());

    dim3 blocks(size / THREADS_PER_BLOCK + (size % THREADS_PER_BLOCK != 0), 1, 1);
    dim3 threads(THREADS_PER_BLOCK, 1, 1);

    bernoulliKernel<<<blocks, threads>>>(primes, size);
    bezierKernel<<<blocks, threads>>>(primes, size);
    bernoulliBezCompositeKernel<<<blocks, threads>>>(primes, size);
    denseBernoulliKernel<<<blocks, threads>>>(primes, size);
    denseBezierKernel<<<blocks, threads>>>(primes, size);
    bernoulliDensePrimeKernel<<<blocks, threads>>>(primes, size);
    bezierDensePrimeKernel<<<blocks, threads>>>(primes, size);
    bernoulliDenseKernel<<<blocks, threads>>>(primes, size);
    bezierDenseKernel<<<blocks, threads>>>(primes, size);
    bernoulliBezierKernel<<<blocks, threads>>>(primes, size);
    bezierBernoulliKernel<<<blocks, threads>>>(primes, size);
    bernoulliDenseBernKernel<<<blocks, threads>>>(primes, size);
    bezierDenseBernKernel<<<blocks, threads>>>(primes, size);
    bernoulliDenseBezierKernel<<<blocks, threads>>>(primes, size);
    bezierDenseBezierKernel<<<blocks, threads>>>(primes, size);
    bernoulliBernKernel<<<blocks, threads>>>(primes, size);
    bezierBernKernel<<<blocks, threads>>>(primes, size);
    bernoulliBezierBernKernel<<<blocks, threads>>>(primes, size);
    bezierBernBezierKernel<<<blocks, threads>>>(primes, size);
    bernoulliDenseBernBezierKernel<<<blocks, threads>>>(primes, size);
    bezierDenseBernBezierKernel<<<blocks, threads>>>(primes, size);

    cudaDeviceSynchronize();

    return 0;
}
