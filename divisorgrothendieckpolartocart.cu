#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

__device__ int find_large_prime(int seed) {
    unsigned long long num = seed * 987654321ULL + 123456789ULL;
    while (!is_prime(num)) {
        num += 2;
    }
    return (int)(num % INT_MAX);
}

__device__ int grothendieck_polartocart_x(int r, float theta) {
    return (int)(r * cosf(theta));
}

__device__ int grothendieck_polartocart_y(int r, float theta) {
    return (int)(r * sinf(theta));
}

__global__ void generate_primes(int* primes, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    primes[idx] = find_large_prime(seed + idx);
}

__global__ void transform_coords(int* x_out, int* y_out, float theta, int r) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    x_out[idx] = grothendieck_polartocart_x(r, theta + idx * 0.1f);
    y_out[idx] = grothendieck_polartocart_y(r, theta + idx * 0.1f);
}

__global__ void random_function_1(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = (idx * 738561297 + 28477774) % INT_MAX;
}

__global__ void random_function_2(float* data, float scalar) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = sinf(idx * 0.3f) * scalar;
}

// ... (16 more similar functions)

int main() {
    const int N = 256;
    int *primes, *x_out, *y_out;
    float *data_float;
    cudaMalloc(&primes, N * sizeof(int));
    cudaMalloc(&x_out, N * sizeof(int));
    cudaMalloc(&y_out, N * sizeof(int));
    cudaMalloc(&data_float, N * sizeof(float));

    int seed = 42;
    generate_primes<<<(N + 255) / 256, 256>>>(primes, seed);
    transform_coords<<<(N + 255) / 256, 256>>>(x_out, y_out, 0.785f, 100);

    // ... (additional kernel launches for random functions)

    cudaMemcpy(primes, primes, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(x_out, x_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_out, y_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_float, data_float, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ... (print or use the results)

    cudaFree(primes);
    cudaFree(x_out);
    cudaFree(y_out);
    cudaFree(data_float);

    return 0;
}
