#include <cuda_runtime.h>
#include <iostream>

__device__ bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (int i = 3; i * i <= num; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

__device__ int denseleibnizdiverg1(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            sum += i;
        }
    }
    return sum;
}

__device__ int denseleibnizdiverg2(int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            count++;
        }
    }
    return count;
}

__device__ int denseleibnizdiverg3(int n) {
    int product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            product *= i;
        }
    }
    return product;
}

__device__ bool denseleibnizdiverg4(int n) {
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            return true;
        }
    }
    return false;
}

__device__ int denseleibnizdiverg5(int n) {
    int max_prime = -1;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            max_prime = i;
        }
    }
    return max_prime;
}

__device__ bool denseleibnizdiverg6(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count > 10;
}

__device__ bool denseleibnizdiverg7(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum > 100;
}

__device__ bool denseleibnizdiverg8(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product > 1000;
}

__device__ bool denseleibnizdiverg9(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count % 2 == 0;
}

__device__ bool denseleibnizdiverg10(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum % 2 == 1;
}

__device__ bool denseleibnizdiverg11(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product % 2 == 0;
}

__device__ bool denseleibnizdiverg12(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count > 5;
}

__device__ bool denseleibnizdiverg13(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum > 50;
}

__device__ bool denseleibnizdiverg14(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product > 500;
}

__device__ bool denseleibnizdiverg15(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count % 3 == 0;
}

__device__ bool denseleibnizdiverg16(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum % 3 == 1;
}

__device__ bool denseleibnizdiverg17(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product % 3 == 2;
}

__device__ bool denseleibnizdiverg18(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count > 2;
}

__device__ bool denseleibnizdiverg19(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum > 20;
}

__device__ bool denseleibnizdiverg20(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product > 200;
}

__device__ bool denseleibnizdiverg21(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count % 4 == 0;
}

__device__ bool denseleibnizdiverg22(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum % 4 == 1;
}

__device__ bool denseleibnizdiverg23(int n) {
    int prime_product = 1;
    for (int i = 2; i < n; ++i) {
        if (isPrime(i)) {
            prime_product *= i;
        }
    }
    return prime_product % 4 == 2;
}

__device__ bool denseleibnizdiverg24(int n) {
    int prime_count = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_count++;
        }
    }
    return prime_count % 5 == 0;
}

__device__ bool denseleibnizdiverg25(int n) {
    int prime_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (isPrime(i)) {
            prime_sum += i;
        }
    }
    return prime_sum % 5 == 1;
}
