#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/random.h>

__global__ void sparseluPrimeCheck(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        bool is_prime = true;
        for (int i = 2; i <= sqrt(d_data[idx]); ++i) {
            if (d_data[idx] % i == 0) {
                is_prime = false;
                break;
            }
        }
        d_is_prime[idx] = is_prime;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseluPrimeSum(int *d_data, int *d_sum) {
    __shared__ int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000000) cache[tid] = d_data[i];
    else cache[tid] = 0;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) cache[tid] += cache[tid + s];
    }

    if (tid == 0) d_sum[blockIdx.x] = cache[0];
}

__global__ void sparsepiPrimeCount(bool *d_is_prime, int *d_count) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000000) d_count[blockIdx.x] = d_is_prime[i];
    else d_count[blockIdx.x] = 0;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) d_count[tid] += d_count[tid + s];
    }

    if (tid == 0) atomicAdd(d_count, cache[0]);
}

__global__ void sparsePhiFunction(int *d_data, int *d_result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        int n = d_data[idx];
        for (int p = 2; p <= sqrt(n); ++p) {
            if (n % p == 0) {
                while (n % p == 0) n /= p;
                d_result[idx] *= (1 - 1.0f / p);
            }
        }
        if (n > 1) d_result[idx] *= (1 - 1.0f / n);
    } else {
        d_result[idx] = 0;
    }
}

__global__ void sparseEulerTotientSum(int *d_data, int *d_sum) {
    __shared__ int cache[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000000) cache[tid] = d_data[i];
    else cache[tid] = 0;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) cache[tid] += cache[tid + s];
    }

    if (tid == 0) d_sum[blockIdx.x] = cache[0];
}

__global__ void sparseSieveOfErathostenes(bool *d_is_prime, int limit) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2) return;

    __syncthreads();

    if (tid == 0 && i <= limit / 2) d_is_prime[i] = false;

    for (unsigned int p = 3; p * p <= limit; p += 2) {
        if (d_is_prime[p]) {
            for (unsigned int j = p * p; j <= limit; j += 2 * p)
                d_is_prime[j] = false;
        }
    }
}

__global__ void sparseMillerRabinTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 3) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0) { d_is_prime[idx] = false; return; }

        unsigned int d = n - 1, s = 0;
        while (!(d & 1)) {
            ++s;
            d >>= 1;
        }

        for (unsigned int i = 0; i < 5; ++i) {
            unsigned int a = 2 + rand() % (n - 4);
            if (!witness(a, s, d, n)) { d_is_prime[idx] = false; return; }
        }
        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparsePollardRhoFactorization(int *d_data, int *d_factors) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int x = rand() % d_data[idx];
        unsigned int y = x;
        unsigned int c = rand() % d_data[idx];
        unsigned int g = 1;

        while (g == 1) {
            x = f(x, c, d_data[idx]);
            y = f(f(y, c, d_data[idx]), c, d_data[idx]);
            g = gcd(abs(x - y), d_data[idx]);
        }

        if (g == d_data[idx]) {
            d_factors[idx] = 0;
        } else {
            d_factors[idx] = g;
        }
    } else {
        d_factors[idx] = 0;
    }
}

__global__ void sparseLucasLehmerTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (!(n & 1)) { d_is_prime[idx] = false; return; }
        unsigned long long s = 4;

        for (unsigned int i = 3; i < n; ++i) {
            s = (s * s - 2) % ((1ULL << n) - 1);
        }

        d_is_prime[idx] = (s == 0);
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseFermatPrimalityTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        unsigned int a = 2;

        while (gcd(a, n) == 1 && mod_exp(a, n - 1, n) == 1) {
            ++a;
            if (a >= n) { d_is_prime[idx] = true; return; }
        }

        d_is_prime[idx] = false;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseCarmichaelFunction(int *d_data, int *d_result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        int result = 1;

        for (int p : get_prime_factors(n)) {
            int e = get_max_exponent(p, n);
            result = lcm(result, phi(pow(p, e)));
        }

        d_result[idx] = result;
    } else {
        d_result[idx] = 0;
    }
}

__global__ void sparseEratosthenesSegmented(int *d_data, bool *d_is_prime) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1000000) d_is_prime[i] = true;
    else return;

    __syncthreads();

    if (tid == 0) d_is_prime[0] = false;
    if (tid == 0) d_is_prime[1] = false;

    for (unsigned int p = 2; p * p <= 1000000; ++p) {
        if (d_is_prime[p]) {
            for (unsigned int j = p * p; j < 1000000; j += p)
                d_is_prime[j] = false;
        }
    }
}

__global__ void sparseAKSPrimalityTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0 || n % 3 == 0) { d_is_prime[idx] = false; return; }

        for (unsigned int i = 5; i * i <= n; ++i += 6)
            if (n % i == 0 || n % (i + 2) == 0) { d_is_prime[idx] = false; return; }

        unsigned int r = get_smallest_r(n);
        for (unsigned int a = 1; a <= sqrtphi(r); ++a)
            if (gcd(a, n) != 1 || !check_cyclotomic_poly(n, a)) { d_is_prime[idx] = false; return; }

        if (!is_smooth_for_r(n - 1, r)) { d_is_prime[idx] = false; return; }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseSieveOfAtkin(int *d_data, bool *d_is_prime) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 1000000) d_is_prime[i] = false;
    else return;

    __syncthreads();

    for (unsigned int x = 1; x * x < 1000000; ++x)
        for (unsigned int y = 1; y * y < 1000000; ++y) {
            unsigned int n = 4 * x * x + y * y;
            if (n <= 1000000 && (n % 12 == 1 || n % 12 == 5))
                d_is_prime[n] ^= true;

            n = 3 * x * x + y * y;
            if (n <= 1000000 && n % 12 == 7)
                d_is_prime[n] ^= true;

            n = 3 * x * x - y * y;
            if (x > y && n <= 1000000 && n % 12 == 11)
                d_is_prime[n] ^= true;
        }

    for (unsigned int i = 5; i * i < 1000000; ++i) {
        if (d_is_prime[i])
            for (unsigned int j = i * i; j < 1000000; j += i * i)
                d_is_prime[j] = false;
    }

    d_is_prime[2] = true;
    d_is_prime[3] = true;
}

__global__ void sparsePocklingtonTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 1 || gcd(n, 60) != 1) { d_is_prime[idx] = false; return; }

        for (unsigned int q : get_factors(60)) {
            if (mod_exp(n - 1, n / q, n) != 1) { d_is_prime[idx] = false; return; }
        }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++k;
        }

        for (unsigned int a = 2; a <= sqrt(n); ++a)
            if (gcd(a, n) == 1 && mod_exp(a, r, n) == 1 && !is_smooth_for_p(n - 1, p)) { d_is_prime[idx] = false; return; }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseWilliamsPrimalityTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4 || gcd(n, 60) != 1) { d_is_prime[idx] = false; return; }

        for (unsigned int a : get_factors(6)) {
            if (mod_exp(a, n - 1, n) != 1) { d_is_prime[idx] = false; return; }
        }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++k;
        }

        for (unsigned int b = 2; b <= sqrt(n); ++b)
            if (gcd(b, n) == 1 && mod_exp(b, r, n) == 1 && !is_smooth_for_p(n - 1, p)) { d_is_prime[idx] = false; return; }

        unsigned int w = 2;
        for (unsigned int c = 3; c <= sqrt(n); ++c)
            if (mod_exp(w, n - 1, n) == 1 && !is_smooth_for_p(n - 1, p)) { d_is_prime[idx] = false; return; }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseMillerRabinTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0 || n % 3 == 0) { d_is_prime[idx] = false; return; }

        for (unsigned int i = 5; i * i <= n; ++i += 6)
            if (n % i == 0 || n % (i + 2) == 0) { d_is_prime[idx] = false; return; }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++s;
        }

        for (unsigned int a : get_random_bases(n)) {
            if (!is_miller_rabin_passed(n, a)) { d_is_prime[idx] = false; return; }
        }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseBailliePSWTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4 || gcd(n, 60) != 1) { d_is_prime[idx] = false; return; }

        for (unsigned int a : get_factors(6)) {
            if (mod_exp(a, n - 1, n) != 1) { d_is_prime[idx] = false; return; }
        }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++k;
        }

        for (unsigned int b = 2; b <= sqrt(n); ++b)
            if (gcd(b, n) == 1 && mod_exp(b, r, n) == 1 && !is_smooth_for_p(n - 1, p)) { d_is_prime[idx] = false; return; }

        unsigned int w = 2;
        for (unsigned int c = 3; c <= sqrt(n); ++c)
            if (mod_exp(w, n - 1, n) == 1 && !is_smooth_for_p(n - 1, p)) { d_is_prime[idx] = false; return; }

        if (!is_miller_rabin_passed(n)) { d_is_prime[idx] = false; return; }
        if (!is_solovay_strassen_passed(n)) { d_is_prime[idx] = false; return; }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseSolovayStrassenTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0 || n % 3 == 0) { d_is_prime[idx] = false; return; }

        for (unsigned int i = 5; i * i <= n; ++i += 6)
            if (n % i == 0 || n % (i + 2) == 0) { d_is_prime[idx] = false; return; }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++s;
        }

        for (unsigned int a : get_random_bases(n)) {
            if (!is_solovay_strassen_passed(n, a)) { d_is_prime[idx] = false; return; }
        }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseFermatTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0 || n % 3 == 0) { d_is_prime[idx] = false; return; }

        for (unsigned int i = 5; i * i <= n; ++i += 6)
            if (n % i == 0 || n % (i + 2) == 0) { d_is_prime[idx] = false; return; }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++s;
        }

        for (unsigned int a : get_random_bases(n)) {
            if (!is_fermat_passed(n, a)) { d_is_prime[idx] = false; return; }
        }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseLucasTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (n <= 4) { d_is_prime[idx] = true; return; }
        if (n % 2 == 0 || n % 3 == 0) { d_is_prime[idx] = false; return; }

        for (unsigned int i = 5; i * i <= n; ++i += 6)
            if (n % i == 0 || n % (i + 2) == 0) { d_is_prime[idx] = false; return; }

        unsigned int r = n - 1;
        while (!(r & 1)) {
            r >>= 1;
            ++s;
        }

        for (unsigned int a : get_random_bases(n)) {
            if (!is_lucas_passed(n, a)) { d_is_prime[idx] = false; return; }
        }

        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}

__global__ void sparseAKSPrimalityTest(int *d_data, bool *d_is_prime) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 1 && d_data[idx] > 1) {
        unsigned int n = d_data[idx];
        if (!is_aks_passed(n)) { d_is_prime[idx] = false; return; }
        d_is_prime[idx] = true;
    } else {
        d_is_prime[idx] = false;
    }
}
