#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ bool boole_is_prime(unsigned long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (unsigned long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

__device__ unsigned long boole_next_prime(unsigned long n) {
    while (!boole_is_prime(n))
        ++n;
    return n;
}

__device__ unsigned long boole_previous_prime(unsigned long n) {
    if (n <= 2) return 0;
    do --n; while (!boole_is_prime(n));
    return n;
}

__device__ bool boole_is_mersenne_prime(unsigned long p) {
    return boole_is_prime((1UL << p) - 1);
}

__device__ unsigned long boole_find_mersenne_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_mersenne_prime(p))
            return (1UL << p) - 1;
    return 0;
}

__device__ bool boole_is_fermat_prime(unsigned long n) {
    unsigned long x = 2;
    return (unsigned long)(pow(x, (double)n) + 1) % n == 0;
}

__device__ unsigned long boole_find_fermat_prime(unsigned long start, unsigned long end) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_fermat_prime(n))
            return n;
    return 0;
}

__device__ bool boole_is_carmichael_number(unsigned long n) {
    for (unsigned long a = 2; a < n; ++a)
        if (__gcd(a, n) == 1 && powmod(a, n - 1, n) != 1)
            return false;
    return true;
}

__device__ unsigned long boole_find_carmichael_number(unsigned long start, unsigned long end) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_carmichael_number(n))
            return n;
    return 0;
}

__device__ bool boole_is_safe_prime(unsigned long p) {
    return boole_is_prime(p) && boole_is_prime((p - 1) / 2);
}

__device__ unsigned long boole_find_safe_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_safe_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_twin_prime(unsigned long p) {
    return boole_is_prime(p) && (boole_is_prime(p - 2) || boole_is_prime(p + 2));
}

__device__ unsigned long boole_find_twin_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_twin_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_lucas_carmichael_number(unsigned long n) {
    for (unsigned long a = 2; a < n; ++a)
        if (__gcd(a, n) == 1 && powmod(a, (n - 1) / 3, n) != 1)
            return false;
    return true;
}

__device__ unsigned long boole_find_lucas_carmichael_number(unsigned long start, unsigned long end) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_lucas_carmichael_number(n))
            return n;
    return 0;
}

__device__ bool boole_is_strong_pseudoprime(unsigned long n, unsigned long a) {
    if (__gcd(a, n) > 1) return false;
    unsigned long d = n - 1, s = 0;
    while (d % 2 == 0) d /= 2, ++s;
    unsigned long x = powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (unsigned long r = 1; r < s; ++r) {
        x = __mulhi(x, x) % n;
        if (x == n - 1) return true;
    }
    return false;
}

__device__ unsigned long boole_find_strong_pseudoprime(unsigned long start, unsigned long end, unsigned long a) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_strong_pseudoprime(n, a))
            return n;
    return 0;
}

__device__ bool boole_is_weak_pseudoprime(unsigned long n, unsigned long a) {
    return powmod(a, n - 1, n) == 1;
}

__device__ unsigned long boole_find_weak_pseudoprime(unsigned long start, unsigned long end, unsigned long a) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_weak_pseudoprime(n, a))
            return n;
    return 0;
}

__device__ bool boole_is_frobenius_pseudoprime(unsigned long n, unsigned long a) {
    if (__gcd(a, n) > 1) return false;
    unsigned long x = powmod(a, (n - 1) / 2, n);
    return (x * x % n == 1) && (a * x * x % n == 1);
}

__device__ unsigned long boole_find_frobenius_pseudoprime(unsigned long start, unsigned long end, unsigned long a) {
    for (unsigned long n = start; n <= end; ++n)
        if (boole_is_frobenius_pseudoprime(n, a))
            return n;
    return 0;
}

__device__ bool boole_is_elliptic_curve_prime(unsigned long p) {
    // Placeholder implementation
    return true;
}

__device__ unsigned long boole_find_elliptic_curve_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_elliptic_curve_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_quadratic_residue(unsigned long a, unsigned long p) {
    return powmod(a, (p - 1) / 2, p) == 1;
}

__device__ unsigned long boole_find_quadratic_residue(unsigned long start, unsigned long end, unsigned long p) {
    for (unsigned long a = start; a <= end; ++a)
        if (boole_is_quadratic_residue(a, p))
            return a;
    return 0;
}

__device__ bool boole_is_quadratic_nonresidue(unsigned long a, unsigned long p) {
    return powmod(a, (p - 1) / 2, p) != 1;
}

__device__ unsigned long boole_find_quadratic_nonresidue(unsigned long start, unsigned long end, unsigned long p) {
    for (unsigned long a = start; a <= end; ++a)
        if (boole_is_quadratic_nonresidue(a, p))
            return a;
    return 0;
}

__device__ bool boole_is_legendre_symbol(unsigned long a, unsigned long p) {
    return powmod(a, (p - 1) / 2, p) == 1 ? 1 : -1;
}

__device__ int boole_find_legendre_symbol(unsigned long start, unsigned long end, unsigned long p) {
    for (unsigned long a = start; a <= end; ++a)
        if (boole_is_legendre_symbol(a, p))
            return a;
    return 0;
}

__device__ bool boole_is_jacobi_symbol(unsigned long a, unsigned long n) {
    int result = 1;
    while (a != 0) {
        if (a % 2 == 0) {
            if (n % 8 == 3 || n % 8 == 5)
                result *= -1;
            a /= 2;
        } else {
            if (a % 4 == 3 && n % 4 == 3)
                result *= -1;
            std::swap(a, n);
        }
    }
    return result != -1;
}

__device__ int boole_find_jacobi_symbol(unsigned long start, unsigned long end, unsigned long n) {
    for (unsigned long a = start; a <= end; ++a)
        if (boole_is_jacobi_symbol(a, n))
            return a;
    return 0;
}

__device__ bool boole_is_kronecker_symbol(unsigned long a, unsigned long n) {
    // Placeholder implementation
    return true;
}

__device__ int boole_find_kronecker_symbol(unsigned long start, unsigned long end, unsigned long n) {
    for (unsigned long a = start; a <= end; ++a)
        if (boole_is_kronecker_symbol(a, n))
            return a;
    return 0;
}

__device__ bool boole_is_totient_prime(unsigned long p) {
    // Placeholder implementation
    return true;
}

__device__ unsigned long boole_find_totient_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_totient_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_carlitz_prime(unsigned long p) {
    // Placeholder implementation
    return true;
}

__device__ unsigned long boole_find_carlitz_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_carlitz_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_bernoulli_prime(unsigned long p) {
    // Placeholder implementation
    return true;
}

__device__ unsigned long boole_find_bernoulli_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_bernoulli_prime(p))
            return p;
    return 0;
}

__device__ bool boole_is_fibonacci_prime(unsigned long p) {
    // Placeholder implementation
    return true;
}

__device__ unsigned long boole_find_fibonacci_prime(unsigned long start, unsigned long end) {
    for (unsigned long p = start; p <= end; ++p)
        if (boole_is_fibonacci_prime(p))
            return p;
    return 0;
}
