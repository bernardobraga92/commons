#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void rationalDirichletComb1(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += (d_primes[tid] * d_primes[tid]) % 37;
}

__global__ void rationalDirichletComb2(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= (d_primes[tid] << 5) | (d_primes[tid] >> 3);
}

__global__ void rationalDirichletComb3(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0xFF) * (d_primes[tid] >> 8)) % 41;
}

__global__ void rationalDirichletComb4(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 7) ^ (d_primes[tid] >> 5));
}

__global__ void rationalDirichletComb5(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 43) * (d_primes[tid] / 47)) % 53;
}

__global__ void rationalDirichletComb6(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 11) | (d_primes[tid] >> 7));
}

__global__ void rationalDirichletComb7(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0xF) * (d_primes[tid] >> 4)) % 59;
}

__global__ void rationalDirichletComb8(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 13) ^ (d_primes[tid] >> 9));
}

__global__ void rationalDirichletComb9(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 61) * (d_primes[tid] / 67)) % 71;
}

__global__ void rationalDirichletComb10(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 17) | (d_primes[tid] >> 13));
}

__global__ void rationalDirichletComb11(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0x7) * (d_primes[tid] >> 3)) % 73;
}

__global__ void rationalDirichletComb12(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 19) ^ (d_primes[tid] >> 15));
}

__global__ void rationalDirichletComb13(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 79) * (d_primes[tid] / 83)) % 89;
}

__global__ void rationalDirichletComb14(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 23) | (d_primes[tid] >> 19));
}

__global__ void rationalDirichletComb15(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0x3) * (d_primes[tid] >> 2)) % 97;
}

__global__ void rationalDirichletComb16(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 25) ^ (d_primes[tid] >> 21));
}

__global__ void rationalDirichletComb17(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 101) * (d_primes[tid] / 103)) % 107;
}

__global__ void rationalDirichletComb18(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 29) | (d_primes[tid] >> 25));
}

__global__ void rationalDirichletComb19(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0x1) * (d_primes[tid] >> 1)) % 109;
}

__global__ void rationalDirichletComb20(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 31) ^ (d_primes[tid] >> 27));
}

__global__ void rationalDirichletComb21(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 113) * (d_primes[tid] / 127)) % 131;
}

__global__ void rationalDirichletComb22(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 3) ^ (d_primes[tid] >> 1));
}

__global__ void rationalDirichletComb23(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0xF0) * (d_primes[tid] >> 4)) % 137;
}

__global__ void rationalDirichletComb24(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 7) | (d_primes[tid] >> 3));
}

__global__ void rationalDirichletComb25(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 139) * (d_primes[tid] / 149)) % 151;
}

__global__ void rationalDirichletComb26(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 11) ^ (d_primes[tid] >> 7));
}

__global__ void rationalDirichletComb27(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0xFF) * (d_primes[tid] >> 8)) % 157;
}

__global__ void rationalDirichletComb28(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 13) | (d_primes[tid] >> 9));
}

__global__ void rationalDirichletComb29(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] % 163) * (d_primes[tid] / 167)) % 173;
}

__global__ void rationalDirichletComb30(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 15) ^ (d_primes[tid] >> 11));
}

__global__ void rationalDirichletComb31(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] += ((d_primes[tid] & 0x7F) * (d_primes[tid] >> 7)) % 179;
}

__global__ void rationalDirichletComb32(unsigned int *d_primes, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_primes[tid] ^= ((d_primes[tid] << 19) | (d_primes[tid] >> 15));
}

int main() {
    const unsigned int N = 1024;
    unsigned int *h_primes = new unsigned int[N];
    for (unsigned int i = 0; i < N; ++i) {
        h_primes[i] = i + 2; // Generate some prime numbers
    }

    unsigned int *d_primes;
    cudaMalloc(&d_primes, N * sizeof(unsigned int));
    cudaMemcpy(d_primes, h_primes, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Call each kernel function in sequence
    rationalDirichletComb0<<<numBlocks, threadsPerBlock>>>(d_primes, N);
    rationalDirichletComb1<<<numBlocks, threadsPerBlock>>>(d_primes, N);
    // ... (call all other kernels)
    rationalDirichletComb32<<<numBlocks, threadsPerBlock>>>(d_primes, N);

    cudaMemcpy(h_primes, d_primes, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_primes);

    delete[] h_primes;
    return 0;
}
