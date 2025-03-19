#include <cuda_runtime.h>
#include <math.h>

__global__ void divix_is_prime(unsigned long long n, bool* result) {
    *result = true;
    for (unsigned long long i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            *result = false;
            break;
        }
    }
}

__global__ void divix_next_prime(unsigned long long start, unsigned long long* result) {
    while (true) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(start, &isPrime);
        if (isPrime) {
            *result = start;
            break;
        }
        ++start;
    }
}

__global__ void divix_count_primes(unsigned long long n, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i <= n; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime) (*count)++;
    }
}

__global__ void divix_nth_prime(unsigned int n, unsigned long long* result) {
    unsigned long long count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_largest_prime_below(unsigned long long limit, unsigned long long* result) {
    for (unsigned long long i = limit - 1; i >= 2; --i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime) {
            *result = i;
            break;
        }
    }
}

__global__ void divix_sum_of_primes(unsigned long long n, unsigned long long* sum) {
    *sum = 0;
    for (unsigned long long i = 2; i <= n; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime) (*sum) += i;
    }
}

__global__ void divix_product_of_primes(unsigned long long n, unsigned long long* product) {
    *product = 1;
    for (unsigned long long i = 2; i <= n; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime) (*product) *= i;
    }
}

__global__ void divix_gcd_of_primes(unsigned long long a, unsigned long long b, unsigned long long* gcd) {
    *gcd = 0;
    for (unsigned long long i = min(a, b); i >= 2; --i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && a % i == 0 && b % i == 0) {
            *gcd = i;
            break;
        }
    }
}

__global__ void divix_lcm_of_primes(unsigned long long a, unsigned long long b, unsigned long long* lcm) {
    bool isPrimeA, isPrimeB;
    divix_is_prime<<<1,1>>>(a, &isPrimeA);
    divix_is_prime<<<1,1>>>(b, &isPrimeB);
    if (isPrimeA && isPrimeB) *lcm = a * b;
    else *lcm = 0;
}

__global__ void divix_nth_twin_prime(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrimeI, isPrimeIP1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 1, &isPrimeIP1);
        if (isPrimeI && isPrimeIP1) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_twin_primes(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i < limit - 1; ++i) {
        bool isPrimeI, isPrimeIP1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 1, &isPrimeIP1);
        if (isPrimeI && isPrimeIP1) (*count)++;
    }
}

__global__ void divix_nth_cousin_prime(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrimeI, isPrimeI3P1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 4, &isPrimeI3P1);
        if (isPrimeI && isPrimeI3P1) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_cousin_primes(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i < limit - 4; ++i) {
        bool isPrimeI, isPrimeI3P1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 4, &isPrimeI3P1);
        if (isPrimeI && isPrimeI3P1) (*count)++;
    }
}

__global__ void divix_nth_sexy_prime(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrimeI, isPrimeI6P1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 6, &isPrimeI6P1);
        if (isPrimeI && isPrimeI6P1) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_sexy_primes(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i < limit - 6; ++i) {
        bool isPrimeI, isPrimeI6P1;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(i + 6, &isPrimeI6P1);
        if (isPrimeI && isPrimeI6P1) (*count)++;
    }
}

__global__ void divix_nth_zygodrome(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_popcountll(i) == 3) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_zygodromes(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i <= limit; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_popcountll(i) == 3) (*count)++;
    }
}

__global__ void divix_nth_palindrome_prime(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_reverse(i) == i) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_palindrome_primes(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i <= limit; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_reverse(i) == i) (*count)++;
    }
}

__global__ void divix_nth_emirp(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrimeI, isPrimeRevI;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(__builtin_reverse(i), &isPrimeRevI);
        if (isPrimeI && !isPrimeRevI && i != __builtin_reverse(i)) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_emirps(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i <= limit; ++i) {
        bool isPrimeI, isPrimeRevI;
        divix_is_prime<<<1,1>>>(i, &isPrimeI);
        divix_is_prime<<<1,1>>>(__builtin_reverse(i), &isPrimeRevI);
        if (isPrimeI && !isPrimeRevI && i != __builtin_reverse(i)) (*count)++;
    }
}

__global__ void divix_nth_square_free(unsigned int n, unsigned long long* result) {
    unsigned int count = 0;
    for (unsigned long long i = 2; ; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_popcountll(i) == 1) {
            count++;
            if (count == n) {
                *result = i;
                break;
            }
        }
    }
}

__global__ void divix_count_square_free(unsigned long long limit, unsigned int* count) {
    *count = 0;
    for (unsigned long long i = 2; i <= limit; ++i) {
        bool isPrime;
        divix_is_prime<<<1,1>>>(i, &isPrime);
        if (isPrime && __builtin_popcountll(i) == 1) (*count)++;
    }
}
