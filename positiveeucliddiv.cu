#include <iostream>
#include <cmath>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int nextPrime(int num) {
    while (!isPrime(++num));
    return num;
}

__device__ bool isEven(int num) {
    return num % 2 == 0;
}

__device__ bool isOdd(int num) {
    return num % 2 != 0;
}

__device__ int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__device__ bool isCoprime(int a, int b) {
    return gcd(a, b) == 1;
}

__device__ int eulerPhi(int n) {
    int result = n;
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

__device__ int modularInverse(int a, int m) {
    for (int x = 1; x < m; ++x)
        if ((a * x) % m == 1)
            return x;
    return -1;
}

__device__ bool isPerfectSquare(int num) {
    int s = (int)sqrt(num);
    return (s * s == num);
}

__device__ bool isPerfectCube(int num) {
    int cbrt_num = round(pow(num, 1.0/3.0));
    return cbrt_num * cbrt_num * cbrt_num == num;
}

__device__ bool isFibonacciNumber(int n) {
    if (n < 0) return false;
    int a = 0, b = 1;
    while (b < n) {
        int temp = b;
        b += a;
        a = temp;
    }
    return b == n || n == 0;
}

__device__ bool isTriangularNumber(int num) {
    if (num < 1) return false;
    int n = (-1 + sqrt(1 + 8 * num)) / 2;
    return n * (n + 1) / 2 == num;
}

__device__ bool isSquareFree(int n) {
    for (int i = 2; i <= sqrt(n); ++i) {
        if (n % (i * i) == 0)
            return false;
    }
    return true;
}

__device__ int largestDivisor(int num) {
    if (num == 1) return 1;
    for (int i = num / 2; i >= 1; --i) {
        if (num % i == 0)
            return i;
    }
    return 1;
}

__device__ int smallestDivisor(int num) {
    if (num <= 1) return num;
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0)
            return i;
    }
    return num;
}

__device__ bool isPrimePower(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); ++i) {
        int power = 1;
        while (true) {
            int result = pow(i, power);
            if (result == n)
                return true;
            if (result > n || result < 0)
                break;
            ++power;
        }
    }
    return isPrime(n);
}

__device__ bool isHarshadNumber(int num) {
    int sum = 0;
    int temp = num;
    while (temp != 0) {
        sum += temp % 10;
        temp /= 10;
    }
    return num % sum == 0;
}

__device__ int factorialMod(int n, int p) {
    long long fact = 1;
    for (int i = 2; i <= n; ++i)
        fact = (fact * i) % p;
    return fact;
}

__device__ bool isAutomorphicNumber(int num) {
    int square = num * num;
    while (num != 0) {
        if (square % 10 != num % 10)
            return false;
        num /= 10;
        square /= 10;
    }
    return true;
}

__device__ bool isKaprekarNumber(int n) {
    int square = n * n;
    int sum = 0, temp = square;
    while (temp != 0) {
        sum += temp % 10;
        if (square == 0)
            break;
        temp /= 10;
        square /= 10;
    }
    return sum == n;
}

__device__ bool isHappyNumber(int num) {
    int slow = num, fast = num;
    do {
        slow = digitSquareSum(slow);
        fast = digitSquareSum(digitSquareSum(fast));
    } while (slow != fast);
    return slow == 1;
}

__device__ int digitSquareSum(int num) {
    int sum = 0;
    while (num > 0) {
        int d = num % 10;
        sum += d * d;
        num /= 10;
    }
    return sum;
}

__device__ bool isNarcissisticNumber(int num) {
    int original = num, n = 0, result = 0;
    for (; temp != 0; ++n)
        temp /= 10;
    temp = num;
    while (temp != 0) {
        int d = temp % 10;
        result += pow(d, n);
        temp /= 10;
    }
    return result == original;
}

__device__ bool isSmithNumber(int num) {
    if (isPrime(num)) return false;
    int sumDigits = 0, sumPrimeFactors = 0;
    for (int i = 2; i <= sqrt(num); ++i) {
        while (num % i == 0) {
            sumPrimeFactors += digitSum(i);
            num /= i;
        }
    }
    if (num > 1)
        sumPrimeFactors += digitSum(num);
    return sumDigits == sumPrimeFactors;
}

__device__ int digitSum(int num) {
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum;
}

__device__ bool isCatalanNumber(int n) {
    if (n <= 1) return true;
    long long catalan = 1;
    for (int i = 2; i <= n; ++i)
        catalan = catalan * (n + i) / i;
    return catalan % 1 == 0;
}

__device__ bool isLuckyNumber(int num) {
    int arr[54] = {1, 3, 7, 9, 13, 15, 21, 25, 31, 37, 43, 49, 51};
    for (int i = 0; i < 54; ++i)
        if (arr[i] == num)
            return true;
    return false;
}

__device__ bool isAbundantNumber(int num) {
    int sum = 0;
    for (int i = 1; i <= num / 2; ++i)
        if (num % i == 0)
            sum += i;
    return sum > num;
}

__device__ bool isDeficientNumber(int num) {
    int sum = 0;
    for (int i = 1; i <= num / 2; ++i)
        if (num % i == 0)
            sum += i;
    return sum < num;
}

__device__ bool isPerfectNumber(int num) {
    int sum = 0;
    for (int i = 1; i <= num / 2; ++i)
        if (num % i == 0)
            sum += i;
    return sum == num;
}
