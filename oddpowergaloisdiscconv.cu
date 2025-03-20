#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void oddPowerGaloisDiscConvKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = ((data[idx] ^ 0x123456789ABCDEFULL) * 17) % 0xFFFFFFFFFFFFFFFFULL;
    }
}

__global__ void galoisFieldMulKernel(unsigned long long *data, int size, unsigned long long multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] * multiplier) % 0xFFFFFFFFFFFFFFFFULL;
    }
}

__global__ void primeCheckKernel(unsigned long long *data, bool *isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        isPrime[idx] = true;
        for (unsigned long long i = 2; i * i <= n; ++i) {
            if (n % i == 0) {
                isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void fermatTestKernel(unsigned long long *data, bool *isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        isPrime[idx] = true;
        for (int i = 0; i < 5; ++i) {
            unsigned long long a = 2 + idx % (n - 4);
            if (__builtin_powll(a, n - 1, n) != 1) {
                isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void millerRabinTestKernel(unsigned long long *data, bool *isPrime, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        isPrime[idx] = true;
        for (int i = 0; i < 5; ++i) {
            unsigned long long a = 2 + idx % (n - 4);
            unsigned long long d = n - 1;
            int s = 0;
            while (d % 2 == 0) {
                d /= 2;
                ++s;
            }
            unsigned long long x = __builtin_powll(a, d, n);
            if (x == 1 || x == n - 1) continue;
            for (int r = 1; r < s; ++r) {
                x = (x * x) % n;
                if (x == 1) {
                    isPrime[idx] = false;
                    break;
                }
                if (x == n - 1) break;
            }
            if (x != n - 1) {
                isPrime[idx] = false;
                break;
            }
        }
    }
}

__global__ void pollardRhoKernel(unsigned long long *data, unsigned long long *factors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        unsigned long long x = 2, y = 2, d = 1;
        while (d == 1) {
            x = ((x * x + idx) % n);
            y = ((y * y + idx) % n);
            y = ((y * y + idx) % n);
            d = __gcd(abs(x - y), n);
        }
        factors[idx] = d;
    }
}

__global__ void eulerTotientKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        for (unsigned long long i = 2; i * i <= n; ++i) {
            if (n % i == 0) {
                while (n % i == 0) n /= i;
                n *= (i - 1);
            }
        }
        if (n > 1) n *= (n - 1);
        data[idx] = n;
    }
}

__global__ void chineseRemainderKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long a = data[idx];
        unsigned long long b = data[(idx + 1) % size];
        unsigned long long m = 1000000007;
        unsigned long long x, y;
        extendedGCD(a, m, x, y);
        data[idx] = (a * x * b + a * b * y) % m;
    }
}

__global__ void extendedGCDKernel(unsigned long long a, unsigned long long b, unsigned long long &x, unsigned long long &y) {
    if (b == 0) {
        x = 1;
        y = 0;
    } else {
        extendedGCDKernel(b, a % b, y, x);
        y -= a / b * x;
    }
}

__global__ void legendreSymbolKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long a = data[idx];
        unsigned long long p = 1000000007;
        unsigned long long result = modPow(a, (p - 1) / 2, p);
        data[idx] = result == 1 ? 1 : (result == p - 1 ? -1 : 0);
    }
}

__global__ void jacobiSymbolKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long a = data[idx];
        unsigned long long n = 1000000007;
        int result = 1;
        while (a != 0) {
            while (a % 2 == 0) {
                a /= 2;
                if ((n % 8) == 3 || (n % 8) == 5) result *= -1;
            }
            swap(a, n);
            if (a % 4 == 3 && n % 4 == 3) result *= -1;
            a %= n;
        }
        data[idx] = n == 1 ? result : 0;
    }
}

__global__ void modPowKernel(unsigned long long base, unsigned long long exp, unsigned long long mod, unsigned long long &result) {
    result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        exp = exp >> 1;
        base = (base * base) % mod;
    }
}

__global__ void carmichaelFunctionKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        for (unsigned long long i = 2; i <= n; ++i) {
            if (__gcd(i, n) == 1) {
                n = lcm(n, phi(i));
            }
        }
        data[idx] = n;
    }
}

__global__ void discreteLogarithmKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long a = 2;
        unsigned long long b = data[idx];
        unsigned long long n = 1000000007;
        unsigned long long result = -1;
        for (unsigned long long x = 0; x < n; ++x) {
            if (__builtin_powll(a, x, n) == b) {
                result = x;
                break;
            }
        }
        data[idx] = result;
    }
}

__global__ void primitiveRootKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        for (unsigned long long g = 2; g < n; ++g) {
            bool isPrimitive = true;
            for (unsigned long long i = 1; i < phi(n); ++i) {
                if (__builtin_powll(g, i, n) == 1) {
                    isPrimitive = false;
                    break;
                }
            }
            if (isPrimitive) {
                data[idx] = g;
                break;
            }
        }
    }
}

__global__ void quadraticResidueKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long a = data[idx];
        unsigned long long p = 1000000007;
        if (a == 0 || legendreSymbol(a, p) == 1) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void eulerTotientKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        for (unsigned long long i = 2; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                while (n % i == 0) {
                    n /= i;
                }
                n *= (i - 1);
            }
        }
        if (n > 1) {
            n *= (n - 1);
        }
        data[idx] = n;
    }
}

__global__ void factorialKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        for (unsigned long long i = 2; i <= n; ++i) {
            n *= i;
        }
        data[idx] = n;
    }
}

__global__ void catalanNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        unsigned long long result = 1;
        for (unsigned long long i = 2; i <= n + 1; ++i) {
            result *= n + i;
        }
        for (unsigned long long i = 2; i <= n; ++i) {
            result /= i;
        }
        data[idx] = result / (n + 1);
    }
}

__global__ void binomialCoefficientKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        unsigned long long k = idx % size;
        unsigned long long result = 1;
        for (unsigned long long i = 0; i < k; ++i) {
            result *= n - i;
        }
        for (unsigned long long i = 2; i <= k; ++i) {
            result /= i;
        }
        data[idx] = result;
    }
}

__global__ void stirlingNumberFirstKindKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        unsigned long long k = idx % size;
        if (n == 0 && k == 0) {
            data[idx] = 1;
        } else if (n == 0 || k == 0) {
            data[idx] = 0;
        } else {
            data[idx] = stirlingNumberFirstKind(n - 1, k - 1) + (n - 1) * stirlingNumberFirstKind(n - 1, k);
        }
    }
}

__global__ void stirlingNumberSecondKindKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        unsigned long long k = idx % size;
        if (n == 0 && k == 0) {
            data[idx] = 1;
        } else if (n == 0 || k == 0) {
            data[idx] = 0;
        } else {
            data[idx] = k * stirlingNumberSecondKind(n - 1, k) + stirlingNumberSecondKind(n - 1, k - 1);
        }
    }
}

__global__ void bernoulliNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> A(n + 1);
        for (unsigned long long m = 0; m <= n; ++m) {
            A[m] = 1 / (m + 1);
            for (int j = m; j > 0; --j) {
                A[j - 1] += j * A[j];
            }
        }
        data[idx] = A[0];
    }
}

__global__ void fibonacciNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        if (n == 0) {
            data[idx] = 0;
        } else if (n == 1) {
            data[idx] = 1;
        } else {
            data[idx] = fibonacciNumber(n - 1) + fibonacciNumber(n - 2);
        }
    }
}

__global__ void lucasNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        if (n == 0) {
            data[idx] = 2;
        } else if (n == 1) {
            data[idx] = 1;
        } else {
            data[idx] = lucasNumber(n - 1) + lucasNumber(n - 2);
        }
    }
}

__global__ void harmonicNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        double result = 0.0;
        for (unsigned long long i = 1; i <= n; ++i) {
            result += 1.0 / i;
        }
        data[idx] = static_cast<unsigned long long>(result);
    }
}

__global__ void mersennePrimeKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long p = data[idx];
        if (isMersennePrime(p)) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void perfectNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> divisors;
        for (unsigned long long i = 1; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                divisors.push_back(i);
                if (i != n / i) {
                    divisors.push_back(n / i);
                }
            }
        }
        unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
        if (sum == n && n != 1) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void abundantNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> divisors;
        for (unsigned long long i = 1; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                divisors.push_back(i);
                if (i != n / i) {
                    divisors.push_back(n / i);
                }
            }
        }
        unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
        if (sum > n) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void deficientNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> divisors;
        for (unsigned long long i = 1; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                divisors.push_back(i);
                if (i != n / i) {
                    divisors.push_back(n / i);
                }
            }
        }
        unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
        if (sum < n) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void amicableNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> divisors;
        for (unsigned long long i = 1; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                divisors.push_back(i);
                if (i != n / i) {
                    divisors.push_back(n / i);
                }
            }
        }
        unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
        vector<unsigned long long> amicableDivisors;
        for (unsigned long long i = 1; i <= sqrt(sum); ++i) {
            if (sum % i == 0) {
                amicableDivisors.push_back(i);
                if (i != sum / i) {
                    amicableDivisors.push_back(sum / i);
                }
            }
        }
        unsigned long long amicableSum = accumulate(amicableDivisors.begin(), amicableDivisors.end(), 0);
        if (amicableSum == n && n != sum) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void sociableNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> sociableChain;
        unsigned long long current = n;
        while (true) {
            vector<unsigned long long> divisors;
            for (unsigned long long i = 1; i <= sqrt(current); ++i) {
                if (current % i == 0) {
                    divisors.push_back(i);
                    if (i != current / i) {
                        divisors.push_back(current / i);
                    }
                }
            }
            unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
            sociableChain.push_back(sum);
            if (sum == n && sociableChain.size() > 1) {
                data[idx] = true;
                break;
            } else if (find(sociableChain.begin(), sociableChain.end(), sum) != sociableChain.end()) {
                data[idx] = false;
                break;
            }
            current = sum;
        }
    }
}

__global__ void happyNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> visited;
        while (n != 1 && find(visited.begin(), visited.end(), n) == visited.end()) {
            visited.push_back(n);
            unsigned long long sumOfSquares = 0;
            while (n > 0) {
                unsigned long long digit = n % 10;
                sumOfSquares += digit * digit;
                n /= 10;
            }
            n = sumOfSquares;
        }
        if (n == 1) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void sadNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> visited;
        while (n != 1 && find(visited.begin(), visited.end(), n) == visited.end()) {
            visited.push_back(n);
            unsigned long long sumOfSquares = 0;
            while (n > 0) {
                unsigned long long digit = n % 10;
                sumOfSquares += digit * digit;
                n /= 10;
            }
            n = sumOfSquares;
        }
        if (n != 1) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void uglyNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        while (n % 2 == 0) {
            n /= 2;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        while (n % 5 == 0) {
            n /= 5;
        }
        if (n == 1) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void superPerfectNumberKernel(unsigned long long *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long n = data[idx];
        vector<unsigned long long> divisors;
        for (unsigned long long i = 1; i <= sqrt(n); ++i) {
            if (n % i == 0) {
                divisors.push_back(i);
                if (i != n / i) {
                    divisors.push_back(n / i);
                }
            }
        }
        unsigned long long sum = accumulate(divisors.begin(), divisors.end(), 0);
        vector<unsigned long long> superDivisors;
        for (unsigned long long i = 1; i <= sqrt(sum); ++i) {
            if (sum % i == 0) {
                superDivisors.push_back(i);
                if (i != sum / i) {
                    superDivisors.push_back(sum / i);
                }
            }
        }
        unsigned long long superSum = accumulate(superDivisors.begin(), superDivisors.end(), 0);
        if (superSum == n) {
            data[idx] = true;
        } else {
            data[idx] = false;
        }
    }
}

__global__ void vampireNumberKernel(unsigned long long *data, size_t length, unsigned int numDigits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        bool isVampire = false;
        for (unsigned int i = 0; i <= (numDigits / 2); ++i) {
            for (unsigned int j = 0; j <= (numDigits / 2); ++j) {
                unsigned long long factor1 = data[idx] / pow(10, numDigits - i);
                unsigned long long factor2 = data[idx] % static_cast<unsigned long long>(pow(10, numDigits - j));
                if (factor1 * factor2 == data[idx]) {
                    isVampire = true;
                    break;
                }
            }
        }
        if (isVampire) {
            data[idx] = 1; // Mark as vampire
        } else {
            data[idx] = 0; // Not a vampire
        }
    }
}

__global__ void palindromicNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long reversed = 0;
        unsigned long long temp = number;
        while (temp > 0) {
            reversed = reversed * 10 + temp % 10;
            temp /= 10;
        }
        if (number == reversed) {
            data[idx] = 1; // Mark as palindromic
        } else {
            data[idx] = 0; // Not a palindromic
        }
    }
}

__global__ void circularPrimeKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        bool isCircularPrime = true;
        unsigned long long number = data[idx];
        unsigned long long numDigits = 0;
        unsigned long long temp = number;
        while (temp > 0) {
            numDigits++;
            temp /= 10;
        }
        for (unsigned int i = 0; i < numDigits && isCircularPrime; ++i) {
            if (!isPrime(number)) {
                isCircularPrime = false;
            }
            number = rotateRight(number, numDigits);
        }
        if (isCircularPrime) {
            data[idx] = 1; // Mark as circular prime
        } else {
            data[idx] = 0; // Not a circular prime
        }
    }
}

__global__ void automorphicNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long square = number * number;
        bool isAutomorphic = true;
        while (number > 0) {
            if (square % 10 != number % 10) {
                isAutomorphic = false;
                break;
            }
            number /= 10;
            square /= 10;
        }
        if (isAutomorphic) {
            data[idx] = 1; // Mark as automorphic
        } else {
            data[idx] = 0; // Not an automorphic
        }
    }
}

__global__ void neonNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long square = number * number;
        unsigned long long sumOfDigits = 0;
        while (square > 0) {
            sumOfDigits += square % 10;
            square /= 10;
        }
        if (sumOfDigits == number) {
            data[idx] = 1; // Mark as neon
        } else {
            data[idx] = 0; // Not a neon
        }
    }
}

__global__ void smithNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDigits = 0;
        unsigned long long temp = number;
        while (temp > 0) {
            sumOfDigits += temp % 10;
            temp /= 10;
        }
        unsigned long long productSum = primeFactorsProductSum(number);
        if (sumOfDigits == productSum) {
            data[idx] = 1; // Mark as smith
        } else {
            data[idx] = 0; // Not a smith
        }
    }
}

__global__ void spyNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDigits = 0;
        unsigned long long productOfDigits = 1;
        while (number > 0) {
            sumOfDigits += number % 10;
            productOfDigits *= number % 10;
            number /= 10;
        }
        if (sumOfDigits == productOfDigits) {
            data[idx] = 1; // Mark as spy
        } else {
            data[idx] = 0; // Not a spy
        }
    }
}

__global__ void harshadNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDigits = 0;
        unsigned long long temp = number;
        while (temp > 0) {
            sumOfDigits += temp % 10;
            temp /= 10;
        }
        if (number % sumOfDigits == 0) {
            data[idx] = 1; // Mark as harshad
        } else {
            data[idx] = 0; // Not a harshad
        }
    }
}

__global__ void happyNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isHappy = true;
        while (number != 1 && number != 4) {
            unsigned long long sumOfSquares = 0;
            unsigned long long temp = number;
            while (temp > 0) {
                sumOfSquares += (temp % 10) * (temp % 10);
                temp /= 10;
            }
            if (sumOfSquares == number) {
                isHappy = false;
                break;
            }
            number = sumOfSquares;
        }
        if (isHappy) {
            data[idx] = 1; // Mark as happy
        } else {
            data[idx] = 0; // Not a happy
        }
    }
}

__global__ void sadNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isSad = false;
        while (number != 1 && number != 4) {
            unsigned long long sumOfSquares = 0;
            unsigned long long temp = number;
            while (temp > 0) {
                sumOfSquares += (temp % 10) * (temp % 10);
                temp /= 10;
            }
            if (sumOfSquares == number) {
                isSad = true;
                break;
            }
            number = sumOfSquares;
        }
        if (isSad) {
            data[idx] = 1; // Mark as sad
        } else {
            data[idx] = 0; // Not a sad
        }
    }
}

__global__ void strongNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfFactorials = 0;
        unsigned long long temp = number;
        while (temp > 0) {
            sumOfFactorials += factorial(temp % 10);
            temp /= 10;
        }
        if (sumOfFactorials == number) {
            data[idx] = 1; // Mark as strong
        } else {
            data[idx] = 0; // Not a strong
        }
    }
}

__global__ void amicableNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length - 1) {
        unsigned long long number1 = data[idx];
        unsigned long long number2 = data[idx + 1];
        unsigned long long sumOfDivisors1 = sumOfDivisors(number1);
        unsigned long long sumOfDivisors2 = sumOfDivisors(number2);
        if (sumOfDivisors1 == number2 && sumOfDivisors2 == number1) {
            data[idx] = 1; // Mark as amicable
            data[idx + 1] = 1;
        } else {
            data[idx] = 0; // Not an amicable pair
            data[idx + 1] = 0;
        }
    }
}

__global__ void perfectNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDivisors = 0;
        for (unsigned long long i = 1; i <= number / 2; ++i) {
            if (number % i == 0) {
                sumOfDivisors += i;
            }
        }
        if (sumOfDivisors == number) {
            data[idx] = 1; // Mark as perfect
        } else {
            data[idx] = 0; // Not a perfect
        }
    }
}

__global__ void abundantNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDivisors = 0;
        for (unsigned long long i = 1; i <= number / 2; ++i) {
            if (number % i == 0) {
                sumOfDivisors += i;
            }
        }
        if (sumOfDivisors > number) {
            data[idx] = 1; // Mark as abundant
        } else {
            data[idx] = 0; // Not an abundant
        }
    }
}

__global__ void deficientNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        unsigned long long sumOfDivisors = 0;
        for (unsigned long long i = 1; i <= number / 2; ++i) {
            if (number % i == 0) {
                sumOfDivisors += i;
            }
        }
        if (sumOfDivisors < number) {
            data[idx] = 1; // Mark as deficient
        } else {
            data[idx] = 0; // Not a deficient
        }
    }
}

__global__ void squareFreeNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isSquareFree = true;
        for (unsigned long long i = 2; i <= sqrt(number); ++i) {
            if (number % (i * i) == 0) {
                isSquareFree = false;
                break;
            }
        }
        if (isSquareFree) {
            data[idx] = 1; // Mark as square free
        } else {
            data[idx] = 0; // Not a square free
        }
    }
}

__global__ void cubeFreeNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isCubeFree = true;
        for (unsigned long long i = 2; i <= cbrt(number); ++i) {
            if (number % (i * i * i) == 0) {
                isCubeFree = false;
                break;
            }
        }
        if (isCubeFree) {
            data[idx] = 1; // Mark as cube free
        } else {
            data[idx] = 0; // Not a cube free
        }
    }
}

__global__ void squareNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isSquare = false;
        for (unsigned long long i = 1; i <= sqrt(number); ++i) {
            if (i * i == number) {
                isSquare = true;
                break;
            }
        }
        if (isSquare) {
            data[idx] = 1; // Mark as square
        } else {
            data[idx] = 0; // Not a square
        }
    }
}

__global__ void cubeNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isCube = false;
        for (unsigned long long i = 1; i <= cbrt(number); ++i) {
            if (i * i * i == number) {
                isCube = true;
                break;
            }
        }
        if (isCube) {
            data[idx] = 1; // Mark as cube
        } else {
            data[idx] = 0; // Not a cube
        }
    }
}

__global__ void triangularNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isTriangular = false;
        for (unsigned long long i = 1; i <= sqrt(2 * number); ++i) {
            if (i * (i + 1) / 2 == number) {
                isTriangular = true;
                break;
            }
        }
        if (isTriangular) {
            data[idx] = 1; // Mark as triangular
        } else {
            data[idx = 0; // Not a triangular
        }
    }
}

__global__ void pentagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isPentagonal = false;
        for (unsigned long long i = 1; i <= sqrt(24 * number + 1); ++i) {
            if (i * (3 * i - 1) / 2 == number) {
                isPentagonal = true;
                break;
            }
        }
        if (isPentagonal) {
            data[idx] = 1; // Mark as pentagonal
        } else {
            data[idx] = 0; // Not a pentagonal
        }
    }
}

__global__ void hexagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isHexagonal = false;
        for (unsigned long long i = 1; i <= sqrt(8 * number + 1); ++i) {
            if (i * (2 * i - 1) == number) {
                isHexagonal = true;
                break;
            }
        }
        if (isHexagonal) {
            data[idx] = 1; // Mark as hexagonal
        } else {
            data[idx] = 0; // Not a hexagonal
        }
    }
}

__global__ void heptagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isHeptagonal = false;
        for (unsigned long long i = 1; i <= sqrt(40 * number + 9); ++i) {
            if (i * (5 * i - 3) / 2 == number) {
                isHeptagonal = true;
                break;
            }
        }
        if (isHeptagonal) {
            data[idx] = 1; // Mark as heptagonal
        } else {
            data[idx] = 0; // Not a heptagonal
        }
    }
}

__global__ void octagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isOctagonal = false;
        for (unsigned long long i = 1; i <= sqrt(3 * number + 1); ++i) {
            if (i * (3 * i - 2) == number) {
                isOctagonal = true;
                break;
            }
        }
        if (isOctagonal) {
            data[idx] = 1; // Mark as octagonal
        } else {
            data[idx] = 0; // Not a octagonal
        }
    }
}

__global__ void nonagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isNonagonal = false;
        for (unsigned long long i = 1; i <= sqrt(5 * number + 4); ++i) {
            if (i * (7 * i - 5) / 2 == number) {
                isNonagonal = true;
                break;
            }
        }
        if (isNonagonal) {
            data[idx] = 1; // Mark as nonagonal
        } else {
            data[idx] = 0; // Not a nonagonal
        }
    }
}

__global__ void decagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isDecagonal = false;
        for (unsigned long long i = 1; i <= sqrt(6 * number + 4); ++i) {
            if (i * (4 * i - 3) == number) {
                isDecagonal = true;
                break;
            }
        }
        if (isDecagonal) {
            data[idx] = 1; // Mark as decagonal
        } else {
            data[idx] = 0; // Not a decagonal
        }
    }
}

__global__ void hendecagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isHendecagonal = false;
        for (unsigned long long i = 1; i <= sqrt(8 * number + 1); ++i) {
            if (i * (9 * i - 7) / 2 == number) {
                isHendecagonal = true;
                break;
            }
        }
        if (isHendecagonal) {
            data[idx] = 1; // Mark as hendecagonal
        } else {
            data[idx] = 0; // Not a hendecagonal
        }
    }
}

__global__ void dodecagonalNumberKernel(unsigned long long *data, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long number = data[idx];
        bool isDodecagonal = false;
        for (unsigned long long i = 1; i <= sqrt(10 * number + 4); ++i) {
            if (i * (5 * i - 4) == number) {
                isDodecagonal = true;
                break;
            }
        }
        if (isDodecagonal) {
            data[idx] = 1; // Mark as dodecagonal
        } else {
            data[idx] = 0; // Not a dodecagonal
        }
    }
}
