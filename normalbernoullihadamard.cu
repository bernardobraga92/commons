#include <iostream>
#include <cmath>

#define N 1024

__global__ void normalDistributionKernel(float* d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.5f;
        float stddev = 0.1f;
        float randomValue = static_cast<float>(rand()) / RAND_MAX;
        d_data[idx] = mean + stddev * sqrt(-2.0f * log(randomValue)) * cos(2.0f * M_PI * randomValue);
    }
}

__global__ void bernoulliKernel(float* d_data, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] = (d_data[idx] > p) ? 1.0f : 0.0f;
    }
}

__global__ void hadamardKernel(float* d_data1, float* d_data2, float* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_result[idx] = d_data1[idx] * d_data2[idx];
    }
}

__global__ void primeCheckKernel(unsigned long* d_numbers, bool* d_isPrime, unsigned long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_numbers[idx] <= limit) {
        for (unsigned long i = 2; i * i <= d_numbers[idx]; ++i) {
            if (d_numbers[idx] % i == 0) {
                d_isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void generatePrimesKernel(unsigned long* d_primes, unsigned long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_primes[idx] <= limit) {
        for (unsigned long i = 2; i <= limit / 2; ++i) {
            if (d_primes[idx] % i == 0 && d_primes[idx] != i) {
                d_primes[idx] += 1;
                --idx;
                break;
            }
        }
    }
}

__global__ void fermatTestKernel(unsigned long* d_numbers, bool* d_isPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned long a = d_numbers[idx] - 1;
        d_isPrime[idx] = (pow(a, d_numbers[idx] / 2) % d_numbers[idx] == 1);
    }
}

__global__ void millerRabinKernel(unsigned long* d_numbers, bool* d_isPrime, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < iterations; ++i) {
            unsigned long a = rand() % (d_numbers[idx] - 4) + 2;
            unsigned long d = d_numbers[idx] - 1;
            int s = 0;
            while ((d & 1) == 0) {
                d >>= 1;
                ++s;
            }
            unsigned long x = pow(a, d) % d_numbers[idx];
            if (x != 1 && x != d_numbers[idx] - 1) {
                for (int r = 1; r < s; ++r) {
                    x = (x * x) % d_numbers[idx];
                    if (x == d_numbers[idx] - 1) break;
                }
                if (x != d_numbers[idx] - 1) {
                    d_isPrime[idx] = false;
                    break;
                }
            }
        }
    }
}

__global__ void pollardRhoKernel(unsigned long* d_numbers, unsigned long* d_factors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned long x = rand() % d_numbers[idx];
        unsigned long y = x;
        unsigned long c = rand() % d_numbers[idx];
        do {
            x = ((x * x) % d_numbers[idx] + c) % d_numbers[idx];
            y = ((y * y) % d_numbers[idx] + c) % d_numbers[idx];
            y = ((y * y) % d_numbers[idx] + c) % d_numbers[idx];
            unsigned long g = gcd(abs(x - y), d_numbers[idx]);
            if (g != 1 && g != d_numbers[idx]) {
                d_factors[idx] = g;
                break;
            }
        } while (g == 1);
    }
}

__global__ void eulerTotientKernel(unsigned long* d_numbers, unsigned long* d_phi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned long n = d_numbers[idx];
        for (unsigned long i = 2; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                while (n % i == 0) n /= i;
                d_phi[idx] *= (i - 1);
            }
        }
        if (n > 1) d_phi[idx] *= (n - 1);
    }
}

__global__ void carmichaelKernel(unsigned long* d_numbers, unsigned long* d_lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned long n = d_numbers[idx];
        for (unsigned long i = 2; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                while (n % i == 0) n /= i;
                d_lambda[idx] = lcm(d_lambda[idx], euler_totient(i));
            }
        }
        if (n > 1) d_lambda[idx] = lcm(d_lambda[idx], euler_totient(n));
    }
}

__global__ void legendreSymbolKernel(unsigned long* d_numbers, unsigned long p, int* d_symbol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_numbers[idx] < p) {
        int a = d_numbers[idx];
        int k = (p - 1) / 2;
        d_symbol[idx] = pow(a, k) % p;
    }
}

__global__ void quadraticResidueKernel(unsigned long* d_numbers, unsigned long p, bool* d_residue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_numbers[idx] < p) {
        for (unsigned long i = 0; i < p; ++i) {
            if ((i * i) % p == d_numbers[idx]) {
                d_residue[idx] = true;
                break;
            }
        }
    }
}

__global__ void chineseRemainderKernel(unsigned long* d_numbers, unsigned long* d_moduli, unsigned long m, unsigned long* d_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_moduli[idx] != 0) {
        unsigned long prod = m / d_moduli[idx];
        unsigned long inverse = modInverse(prod, d_moduli[idx]);
        d_result[idx] = (d_numbers[idx] * prod * inverse) % m;
    }
}

__global__ void extendedEuclideanKernel(unsigned long a, unsigned long b, unsigned long* d_gcd, long* d_x, long* d_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        if (a == 0) {
            d_gcd[0] = b;
            d_x[0] = 0;
            d_y[0] = 1;
        } else {
            long x1, y1;
            extendedEuclidean(b % a, a, d_gcd, &x1, &y1);
            d_gcd[0] = d_gcd[0];
            d_x[0] = y1 - (b / a) * x1;
            d_y[0] = x1;
        }
    }
}

__global__ void sieveOfEratosthenesKernel(bool* d_isPrime, unsigned long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx >= 2) {
        for (unsigned long i = 2; i <= sqrt(limit); ++i) {
            if (d_isPrime[i] && idx % i == 0) {
                d_isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void lucasTestKernel(unsigned long* d_numbers, bool* d_isPrime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned long p = d_numbers[idx];
        unsigned long P = 1, Q = -1, D = P * P - 4 * Q;
        unsigned long U = 0, V = 2;
        while (p > 1) {
            if (p % 2 == 1) {
                long nextU = U * V - D * U / p;
                long nextV = V * V - 2 * Q;
                U = nextU;
                V = nextV;
                p -= 1;
            } else {
                long nextU = (V * V - 2) / p;
                long nextV = 2 * V * P - V;
                U = nextU;
                V = nextV;
                p /= 2;
            }
        }
        d_isPrime[idx] = (U == 0);
    }
}

__global__ void millerRabinTestKernel(unsigned long* d_numbers, unsigned long iterations, bool* d_prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && d_numbers[idx] > 1) {
        for (unsigned long i = 0; i < iterations; ++i) {
            unsigned long a = 2 + rand() % (d_numbers[idx] - 4);
            if (!millerRabinTest(d_numbers[idx], a)) {
                d_prime[idx] = false;
                break;
            }
        }
    } else {
        d_prime[idx] = true;
    }
}

__global__ void akSieveKernel(unsigned long* d_numbers, unsigned long limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx >= 2) {
        for (unsigned long i = 2; i <= sqrt(limit); ++i) {
            if (d_numbers[i] != 0 && d_numbers[idx] % d_numbers[i] == 0) {
                d_numbers[idx] = 0;
                break;
            }
        }
    }
}

__global__ void fibonacciKernel(unsigned long* d_numbers, unsigned long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx <= n) {
        if (idx == 0 || idx == 1) {
            d_numbers[idx] = idx;
        } else {
            d_numbers[idx] = d_numbers[idx - 1] + d_numbers[idx - 2];
        }
    }
}

__global__ void factorialKernel(unsigned long* d_numbers, unsigned long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx <= n) {
        d_numbers[idx] = 1;
        for (unsigned long i = 2; i <= idx; ++i) {
            d_numbers[idx] *= i;
        }
    }
}

__global__ void gcdKernel(unsigned long* d_numbers, unsigned long* d_gcd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx > 0) {
        d_gcd[0] = gcd(d_numbers[0], d_numbers[idx]);
    }
}

__global__ void lcmKernel(unsigned long* d_numbers, unsigned long* d_lcm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx > 0) {
        d_lcm[0] = lcm(d_numbers[0], d_numbers[idx]);
    }
}

__global__ void modInverseKernel(unsigned long a, unsigned long m, unsigned long* d_inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        d_inverse[0] = modInverse(a, m);
    }
}

int main() {
    // Example usage of the kernels
    return 0;
}
