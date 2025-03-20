#include <cuda_runtime.h>
#include <math.h>

__device__ bool isPrime(unsigned int n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (unsigned int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

__device__ unsigned int generateRandomPrime(unsigned int seed) {
    unsigned int candidate = seed * seed + seed + 41;
    while (!isPrime(candidate)) {
        candidate++;
    }
    return candidate;
}

__global__ void findPrimes(unsigned int *d_primes, unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_primes[idx] = generateRandomPrime(seed + idx);
}

__global__ void denseDescartesFourierCoefficients(float *coeffs, unsigned int prime, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] = sinf((float)(idx * prime) / n) * cosf((float)(idx * prime) / n);
    }
}

__global__ void filterPrimes(unsigned int *d_primes, bool *d_isPrime, unsigned int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_isPrime[idx] = isPrime(d_primes[idx]);
    }
}

__global__ void multiplyCoeffs(float *coeffs1, float *coeffs2, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = coeffs1[idx] * coeffs2[idx];
    }
}

__global__ void addCoeffs(float *coeffs1, float *coeffs2, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = coeffs1[idx] + coeffs2[idx];
    }
}

__global__ void subtractCoeffs(float *coeffs1, float *coeffs2, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = coeffs1[idx] - coeffs2[idx];
    }
}

__global__ void normalizeCoeffs(float *coeffs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] /= n;
    }
}

__global__ void reverseCoeffs(float *coeffs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n / 2) {
        float temp = coeffs[idx];
        coeffs[idx] = coeffs[n - idx - 1];
        coeffs[n - idx - 1] = temp;
    }
}

__global__ void computeInverseFFT(float *coeffs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] = sinf((float)(idx * idx) / n) + cosf((float)(idx * idx) / n);
    }
}

__global__ void computeFFT(float *coeffs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] = sinf((float)(idx * idx) / n) - cosf((float)(idx * idx) / n);
    }
}

__global__ void applyWindowFunction(float *coeffs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] *= sinf((float)idx / n);
    }
}

__global__ void computeDerivativeCoefficients(float *coeffs, float *derivatives, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n - 1) {
        derivatives[idx] = (coeffs[idx + 1] - coeffs[idx]) / 2.0f;
    }
}

__global__ void integrateCoefficients(float *coeffs, float *integrand, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n - 1) {
        integrand[idx] = (coeffs[idx + 1] + coeffs[idx]) / 2.0f;
    }
}

__global__ void scaleCoefficients(float *coeffs, float factor, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] *= factor;
    }
}

__global__ void shiftCoefficients(float *coeffs, float offset, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        coeffs[idx] += offset;
    }
}

int main() {
    const unsigned int numPrimes = 256;
    const unsigned int primeSeed = 12345;
    const int nCoeffs = 1024;

    unsigned int *d_primes;
    cudaMalloc(&d_primes, numPrimes * sizeof(unsigned int));
    findPrimes<<<(numPrimes + 255) / 256, 256>>>(d_primes, primeSeed);

    float *d_coeffs1, *d_coeffs2, *d_result;
    cudaMalloc(&d_coeffs1, nCoeffs * sizeof(float));
    cudaMalloc(&d_coeffs2, nCoeffs * sizeof(float));
    cudaMalloc(&d_result, nCoeffs * sizeof(float));

    unsigned int prime;
    cudaMemcpy(&prime, d_primes + blockIdx.x, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    denseDescartesFourierCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, prime, nCoeffs);
    denseDescartesFourierCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs2, prime + 1, nCoeffs);

    multiplyCoeffs<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, d_coeffs2, d_result, nCoeffs);
    addCoeffs<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, d_coeffs2, d_result, nCoeffs);
    subtractCoeffs<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, d_coeffs2, d_result, nCoeffs);
    normalizeCoeffs<<<(nCoeffs + 255) / 256, 256>>>(d_result, nCoeffs);

    reverseCoeffs<<<(nCoeffs / 2 + 255) / 256, 256>>>(d_coeffs1, nCoeffs);
    computeInverseFFT<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, nCoeffs);
    computeFFT<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, nCoeffs);

    applyWindowFunction<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, nCoeffs);
    computeDerivativeCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, d_result, nCoeffs);
    integrateCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, d_result, nCoeffs);

    scaleCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, 2.0f, nCoeffs);
    shiftCoefficients<<<(nCoeffs + 255) / 256, 256>>>(d_coeffs1, 3.0f, nCoeffs);

    cudaFree(d_primes);
    cudaFree(d_coeffs1);
    cudaFree(d_coeffs2);
    cudaFree(d_result);

    return 0;
}
