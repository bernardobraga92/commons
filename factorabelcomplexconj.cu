#include <iostream>
#include <cmath>

__global__ void complexConjugate(float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = conjf(a[idx]);
    }
}

__global__ void isPrime(unsigned long long* num, bool* result) {
    unsigned long long val = num[0];
    if (val <= 1) { result[0] = false; return; }
    if (val == 2 || val == 3) { result[0] = true; return; }
    if (val % 2 == 0 || val % 3 == 0) { result[0] = false; return; }
    for (unsigned long long i = 5; i * i <= val; i += 6) {
        if (val % i == 0 || val % (i + 2) == 0) { result[0] = false; return; }
    }
    result[0] = true;
}

__global__ void generatePrimes(unsigned long long* primes, int n) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long i = 2; ; ++i) {
            bool is_prime = true;
            for (unsigned long long j = 2; j <= sqrt(i); ++j) {
                if (i % j == 0) { is_prime = false; break; }
            }
            if (is_prime) {
                primes[idx] = i;
                break;
            }
        }
    }
}

__global__ void nextPrime(unsigned long long* current, unsigned long long* next) {
    unsigned long long start = atomicAdd(current, 1);
    for (unsigned long long i = start; ; ++i) {
        bool is_prime = true;
        for (unsigned long long j = 2; j <= sqrt(i); ++j) {
            if (i % j == 0) { is_prime = false; break; }
        }
        if (is_prime) {
            next[0] = i;
            break;
        }
    }
}

__global__ void primeFactorization(unsigned long long* number, unsigned long long* factors, int* factorCount) {
    unsigned long long num = number[0];
    int count = 0;
    for (unsigned long long i = 2; i <= sqrt(num); ++i) {
        while (num % i == 0) {
            factors[count++] = i;
            num /= i;
        }
    }
    if (num > 1) factors[count++] = num;
    factorCount[0] = count;
}

__global__ void sumOfPrimes(unsigned long long* primes, int n, unsigned long long* sum) {
    __shared__ unsigned long long shared_sum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        shared_sum[threadIdx.x] = primes[idx];
    } else {
        shared_sum[threadIdx.x] = 0;
    }
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0) atomicAdd(sum, shared_sum[0]);
}

__global__ void largestPrime(unsigned long long* numbers, int n, unsigned long long* maxPrime) {
    __shared__ unsigned long long shared_max[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_prime = true;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) { is_prime = false; break; }
        }
        shared_max[threadIdx.x] = is_prime ? numbers[idx] : 0;
    } else {
        shared_max[threadIdx.x] = 0;
    }
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s && shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0) atomicMax(maxPrime, shared_max[0]);
}

__global__ void primeCount(unsigned long long* numbers, int n, int* count) {
    __shared__ int shared_count[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_prime = true;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) { is_prime = false; break; }
        }
        shared_count[threadIdx.x] = is_prime ? 1 : 0;
    } else {
        shared_count[threadIdx.x] = 0;
    }
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            shared_count[threadIdx.x] += shared_count[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0) atomicAdd(count, shared_count[0]);
}

__global__ void twinPrimes(unsigned long long* numbers, int n, unsigned long long* twins) {
    __shared__ unsigned long long shared_twins[256][2];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        bool is_prime1 = true, is_prime2 = true;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) { is_prime1 = false; break; }
        }
        for (unsigned long long j = 2; j <= sqrt(numbers[idx + 1]); ++j) {
            if (numbers[idx + 1] % j == 0) { is_prime2 = false; break; }
        }
        if (is_prime1 && is_prime2 && numbers[idx + 1] - numbers[idx] == 2) {
            shared_twins[threadIdx.x][0] = numbers[idx];
            shared_twins[threadIdx.x][1] = numbers[idx + 1];
        } else {
            shared_twins[threadIdx.x][0] = 0;
            shared_twins[threadIdx.x][1] = 0;
        }
    } else {
        shared_twins[threadIdx.x][0] = 0;
        shared_twins[threadIdx.x][1] = 0;
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            if (shared_twins[i][0] != 0 && shared_twins[i][1] != 0) {
                twins[2 * idx + 0] = shared_twins[i][0];
                twins[2 * idx + 1] = shared_twins[i][1];
            }
        }
    }
}

__global__ void primeGaps(unsigned long long* numbers, int n, unsigned long long* gaps) {
    __shared__ unsigned long long shared_gaps[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        bool is_prime1 = true, is_prime2 = true;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) { is_prime1 = false; break; }
        }
        for (unsigned long long j = 2; j <= sqrt(numbers[idx + 1]); ++j) {
            if (numbers[idx + 1] % j == 0) { is_prime2 = false; break; }
        }
        if (is_prime1 && is_prime2) {
            shared_gaps[threadIdx.x] = numbers[idx + 1] - numbers[idx];
        } else {
            shared_gaps[threadIdx.x] = 0;
        }
    } else {
        shared_gaps[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            if (shared_gaps[i] != 0) {
                gaps[idx] = shared_gaps[i];
            }
        }
    }
}

__global__ void primeSieve(unsigned long long* numbers, int n, bool* isPrimeArray) {
    __shared__ bool shared_is_prime[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                shared_is_prime[threadIdx.x] = false;
                break;
            }
        }
        if (shared_is_prime[threadIdx.x]) {
            isPrimeArray[idx] = true;
        } else {
            isPrimeArray[idx] = false;
        }
    }
}

__global__ void primeMultiples(unsigned long long* numbers, int n, unsigned long long* multiples) {
    __shared__ unsigned long long shared_multiples[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                shared_multiples[threadIdx.x] = numbers[idx];
                break;
            }
        }
        multiples[idx] = shared_multiples[threadIdx.x];
    }
}

__global__ void primeDivisors(unsigned long long* numbers, int n, unsigned long long* divisors) {
    __shared__ unsigned long long shared_divisors[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                shared_divisors[threadIdx.x] = j;
                break;
            }
        }
        divisors[idx] = shared_divisors[threadIdx.x];
    }
}

__global__ void primeFibonacci(unsigned long long* numbers, int n, bool* isFibonacciPrime) {
    __shared__ bool shared_is_fib_prime[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long a = 0, b = 1, c;
        while (c < numbers[idx]) {
            c = a + b;
            a = b;
            b = c;
        }
        if (c == numbers[idx]) {
            bool is_prime = true;
            for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
                if (numbers[idx] % j == 0) {
                    is_prime = false;
                    break;
                }
            }
            shared_is_fib_prime[threadIdx.x] = is_prime;
        } else {
            shared_is_fib_prime[threadIdx.x] = false;
        }
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            if (shared_is_fib_prime[i]) {
                isFibonacciPrime[idx + i] = true;
            } else {
                isFibonacciPrime[idx + i] = false;
            }
        }
    }
}

__global__ void primePalindrome(unsigned long long* numbers, int n, bool* isPalindromePrime) {
    __shared__ bool shared_is_palindrome_prime[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_prime = true;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            unsigned long long num = numbers[idx], reversed_num = 0, original = numbers[idx];
            while (num > 0) {
                reversed_num = reversed_num * 10 + num % 10;
                num /= 10;
            }
            if (original == reversed_num) {
                shared_is_palindrome_prime[threadIdx.x] = true;
            } else {
                shared_is_palindrome_prime[threadIdx.x] = false;
            }
        } else {
            shared_is_palindrome_prime[threadIdx.x] = false;
        }
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            if (shared_is_palindrome_prime[i]) {
                isPalindromePrime[idx + i] = true;
            } else {
                isPalindromePrime[idx + i] = false;
            }
        }
    }
}

__global__ void primePower(unsigned long long* numbers, int n, unsigned long long* powers) {
    __shared__ unsigned long long shared_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                unsigned long long power = 1;
                while (numbers[idx] % j == 0) {
                    numbers[idx] /= j;
                    power++;
                }
                shared_powers[threadIdx.x] = power;
                break;
            }
        }
        powers[idx] = shared_powers[threadIdx.x];
    }
}

__global__ void primeFactor(unsigned long long* numbers, int n, unsigned long long* factors) {
    __shared__ unsigned long long shared_factors[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                shared_factors[threadIdx.x] = j;
                break;
            }
        }
        factors[idx] = shared_factors[threadIdx.x];
    }
}

__global__ void primeRoot(unsigned long long* numbers, int n, unsigned long long* roots) {
    __shared__ unsigned long long shared_roots[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                unsigned long long root = pow(j, 1.0 / 3);
                shared_roots[threadIdx.x] = root;
                break;
            }
        }
        roots[idx] = shared_roots[threadIdx.x];
    }
}

__global__ void primeSum(unsigned long long* numbers, int n, unsigned long long* sums) {
    __shared__ unsigned long long shared_sums[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j;
            }
        }
        shared_sums[threadIdx.x] = sum;
    }
    sums[idx] = shared_sums[threadIdx.x];
}

__global__ void primeProduct(unsigned long long* numbers, int n, unsigned long long* products) {
    __shared__ unsigned long long shared_products[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j;
            }
        }
        shared_products[threadIdx.x] = product;
    }
    products[idx] = shared_products[threadIdx.x];
}

__global__ void primeCount(unsigned long long* numbers, int n, unsigned long long* counts) {
    __shared__ unsigned long long shared_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long count = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                count++;
            }
        }
        shared_counts[threadIdx.x] = count;
    }
    counts[idx] = shared_counts[threadIdx.x];
}

__global__ void primeDistinctCount(unsigned long long* numbers, int n, unsigned long long* distinctCounts) {
    __shared__ unsigned long long shared_distinct_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        std::set<unsigned long long> divisors;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                divisors.insert(j);
            }
        }
        shared_distinct_counts[threadIdx.x] = divisors.size();
    }
    distinctCounts[idx] = shared_distinct_counts[threadIdx.x];
}

__global__ void primeEvenCount(unsigned long long* numbers, int n, unsigned long long* evenCounts) {
    __shared__ unsigned long long shared_even_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long even_count = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0 && j % 2 == 0) {
                even_count++;
            }
        }
        shared_even_counts[threadIdx.x] = even_count;
    }
    evenCounts[idx] = shared_even_counts[threadIdx.x];
}

__global__ void primeOddCount(unsigned long long* numbers, int n, unsigned long long* oddCounts) {
    __shared__ unsigned long long shared_odd_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long odd_count = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0 && j % 2 != 0) {
                odd_count++;
            }
        }
        shared_odd_counts[threadIdx.x = odd_count;
    }
}

__global__ void primeEvenDistinctCount(unsigned long long* numbers, int n, unsigned long long* evenDistinctCounts) {
    __shared__ unsigned long long shared_even_distinct_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        std::set<unsigned long long> divisors;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0 && j % 2 == 0) {
                divisors.insert(j);
            }
        }
        shared_even_distinct_counts[threadIdx.x] = divisors.size();
    }
    evenDistinctCounts[idx] = shared_even_distinct_counts[threadIdx.x];
}

__global__ void primeOddDistinctCount(unsigned long long* numbers, int n, unsigned long long* oddDistinctCounts) {
    __shared__ unsigned long long shared_odd_distinct_counts[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        std::set<unsigned long long> divisors;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0 && j % 2 != 0) {
                divisors.insert(j);
            }
        }
        shared_odd_distinct_counts[threadIdx.x] = divisors.size();
    }
    oddDistinctCounts[idx] = shared_odd_distinct_counts[threadIdx.x];
}

__global__ void primeSumOfSquares(unsigned long long* numbers, int n, unsigned long long* sumOfSquares) {
    __shared__ unsigned long long shared_sum_of_squares[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j;
            }
        }
        shared_sum_of_squares[threadIdx.x] = sum;
    }
    sumOfSquares[idx] = shared_sum_of_squares[threadIdx.x];
}

__global__ void primeProductOfSquares(unsigned long long* numbers, int n, unsigned long long* productOfSquares) {
    __shared__ unsigned long long shared_product_of_squares[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j;
            }
        }
        shared_product_of_squares[threadIdx.x] = product;
    }
    productOfSquares[idx] = shared_product_of_squares[threadIdx.x];
}

__global__ void primeSumOfCubes(unsigned long long* numbers, int n, unsigned long long* sumOfCubes) {
    __shared__ unsigned long long shared_sum_of_cubes[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j;
            }
        }
        shared_sum_of_cubes[threadIdx.x] = sum;
    }
    sumOfCubes[idx] = shared_sum_of_cubes[threadIdx.x];
}

__global__ void primeProductOfCubes(unsigned long long* numbers, int n, unsigned long long* productOfCubes) {
    __shared__ unsigned long long shared_product_of_cubes[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j;
            }
        }
        shared_product_of_cubes[threadIdx.x] = product;
    }
    productOfCubes[idx] = shared_product_of_cubes[threadIdx.x];
}

__global__ void primeSumOfFourthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfFourthPowers) {
    __shared__ unsigned long long shared_sum_of_fourth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j;
            }
        }
        shared_sum_of_fourth_powers[threadIdx.x] = sum;
    }
    sumOfFourthPowers[idx] = shared_sum_of_fourth_powers[threadIdx.x];
}

__global__ void primeProductOfFourthPowers(unsigned long long* numbers, int n, unsigned long long* productOfFourthPowers) {
    __shared__ unsigned long long shared_product_of_fourth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j;
            }
        }
        shared_product_of_fourth_powers[threadIdx.x] = product;
    }
    productOfFourthPowers[idx] = shared_product_of_fourth_powers[threadIdx.x];
}

__global__ void primeSumOfFifthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfFifthPowers) {
    __shared__ unsigned long long shared_sum_of_fifth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j;
            }
        }
        shared_sum_of_fifth_powers[threadIdx.x] = sum;
    }
    sumOfFifthPowers[idx] = shared_sum_of_fifth_powers[threadIdx.x];
}

__global__ void primeProductOfFifthPowers(unsigned long long* numbers, int n, unsigned long long* productOfFifthPowers) {
    __shared__ unsigned long long shared_product_of_fifth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j;
            }
        }
        shared_product_of_fifth_powers[threadIdx.x] = product;
    }
    productOfFifthPowers[idx] = shared_product_of_fifth_powers[threadIdx.x];
}

__global__ void primeSumOfSixthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfSixthPowers) {
    __shared__ unsigned long long shared_sum_of_sixth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j * j;
            }
        }
        shared_sum_of_sixth_powers[threadIdx.x] = sum;
    }
    sumOfSixthPowers[idx] = shared_sum_of_sixth_powers[threadIdx.x];
}

__global__ void primeProductOfSixthPowers(unsigned long long* numbers, int n, unsigned long long* productOfSixthPowers) {
    __shared__ unsigned long long shared_product_of_sixth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j * j;
            }
        }
        shared_product_of_sixth_powers[threadIdx.x] = product;
    }
    productOfSixthPowers[idx] = shared_product_of_sixth_powers[threadIdx.x];
}

__global__ void primeSumOfSeventhPowers(unsigned long long* numbers, int n, unsigned long long* sumOfSeventhPowers) {
    __shared__ unsigned long long shared_sum_of_seventh_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j * j * j;
            }
        }
        shared_sum_of_seventh_powers[threadIdx.x] = sum;
    }
    sumOfSeventhPowers[idx] = shared_sum_of_seventh_powers[threadIdx.x];
}

__global__ void primeProductOfSeventhPowers(unsigned long long* numbers, int n, unsigned long long* productOfSeventhPowers) {
    __shared__ unsigned long long shared_product_of_seventh_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j * j * j;
            }
        }
        shared_product_of_seventh_powers[threadIdx.x] = product;
    }
    productOfSeventhPowers[idx] = shared_product_of_seventh_powers[threadIdx.x];
}

__global__ void primeSumOfEighthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfEighthPowers) {
    __shared__ unsigned long long shared_sum_of_eighth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j * j * j * j;
            }
        }
        shared_sum_of_eighth_powers[threadIdx.x] = sum;
    }
    sumOfEighthPowers[idx] = shared_sum_of_eighth_powers[threadIdx.x];
}

__global__ void primeProductOfEighthPowers(unsigned long long* numbers, int n, unsigned long long* productOfEighthPowers) {
    __shared__ unsigned long long shared_product_of_eighth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j * j * j * j;
            }
        }
        shared_product_of_eighth_powers[threadIdx.x] = product;
    }
    productOfEighthPowers[idx] = shared_product_of_eighth_powers[threadIdx.x];
}

__global__ void primeSumOfNinthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfNinthPowers) {
    __shared__ unsigned long long shared_sum_of_ninth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j * j * j * j * j;
            }
        }
        shared_sum_of_ninth_powers[threadIdx.x] = sum;
    }
    sumOfNinthPowers[idx] = shared_sum_of_ninth_powers[threadIdx.x];
}

__global__ void primeProductOfNinthPowers(unsigned long long* numbers, int n, unsigned long long* productOfNinthPowers) {
    __shared__ unsigned long long shared_product_of_ninth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j * j * j * j * j;
            }
        }
        shared_product_of_ninth_powers[threadIdx.x] = product;
    }
    productOfNinthPowers[idx] = shared_product_of_ninth_powers[threadIdx.x];
}

__global__ void primeSumOfTenthPowers(unsigned long long* numbers, int n, unsigned long long* sumOfTenthPowers) {
    __shared__ unsigned long long shared_sum_of_tenth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long sum = 0;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                sum += j * j * j * j * j * j * j * j * j * j;
            }
        }
        shared_sum_of_tenth_powers[threadIdx.x] = sum;
    }
    sumOfTenthPowers[idx] = shared_sum_of_tenth_powers[threadIdx.x];
}

__global__ void primeProductOfTenthPowers(unsigned long long* numbers, int n, unsigned long long* productOfTenthPowers) {
    __shared__ unsigned long long shared_product_of_tenth_powers[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long product = 1;
        for (unsigned long long j = 2; j <= sqrt(numbers[idx]); ++j) {
            if (numbers[idx] % j == 0) {
                product *= j * j * j * j * j * j * j * j * j * j;
            }
        }
        shared_product_of_tenth_powers[threadIdx.x] = product;
    }
    productOfTenthPowers[idx] = shared_product_of_tenth_powers[threadIdx.x];
}

int main() {
    // Example usage of the kernels
    unsigned long long numbers[256] = { /* initialize with some values */ };
    int n = 256;

    unsigned long long* d_numbers;
    cudaMalloc((void**)&d_numbers, n * sizeof(unsigned long long));
    cudaMemcpy(d_numbers, numbers, n * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    unsigned long long sumOfFirstPowers[256];
    unsigned long long productOfFirstPowers[256];

    primeSumOfFirstPowers<<<1, 256>>>(d_numbers, n, sumOfFirstPowers);
    primeProductOfFirstPowers<<<1, 256>>>(d_numbers, n, productOfFirstPowers);

    cudaMemcpy(sumOfFirstPowers, d_numbers, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(productOfFirstPowers, d_numbers, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_numbers);

    // Output the results
    for (int i = 0; i < n; ++i) {
        printf("Sum of first powers: %llu\n", sumOfFirstPowers[i]);
        printf("Product of first powers: %llu\n", productOfFirstPowers[i]);
    }

    return 0;
}
