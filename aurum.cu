#include <cuda_runtime.h>
#include <iostream>

#define DEVICE __device__
#define HOST_DEVICE __host__ __device__

HOST_DEVICE bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}

DEVICE int aurum_find_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isAurumPrime(int num) {
    if (!isPrime(num)) return false;
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return isPrime(sum);
}

DEVICE int aurum_special_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isMegaAurumPrime(int num) {
    if (!isAurumPrime(num)) return false;
    int digitCount = 0;
    while (num > 0) {
        ++digitCount;
        num /= 10;
    }
    return digitCount % 2 == 1;
}

DEVICE int aurum_mega_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isMegaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isGigaAurumPrime(int num) {
    if (!isMegaAurumPrime(num)) return false;
    int reversed = 0;
    int original = num;
    while (num > 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return original == reversed;
}

DEVICE int aurum_giga_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isGigaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isTeraAurumPrime(int num) {
    if (!isGigaAurumPrime(num)) return false;
    int primeCount = 0;
    for (int i = 2; i <= num; ++i) {
        if (isPrime(i)) ++primeCount;
    }
    return isPrime(primeCount);
}

DEVICE int aurum_tera_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isTeraAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isPetaAurumPrime(int num) {
    if (!isTeraAurumPrime(num)) return false;
    int powerSum = 0;
    while (num > 0) {
        int digit = num % 10;
        powerSum += digit * digit;
        num /= 10;
    }
    return isPrime(powerSum);
}

DEVICE int aurum_peta_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isPetaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isExaAurumPrime(int num) {
    if (!isPetaAurumPrime(num)) return false;
    int product = 1;
    while (num > 0) {
        product *= num % 10;
        num /= 10;
    }
    return isPrime(product);
}

DEVICE int aurum_exa_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isExaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isZettaAurumPrime(int num) {
    if (!isExaAurumPrime(num)) return false;
    int digitProduct = 1;
    while (num > 0) {
        digitProduct *= num % 10;
        num /= 10;
    }
    return isPrime(digitProduct);
}

DEVICE int aurum_zetta_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isZettaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isYottaAurumPrime(int num) {
    if (!isZettaAurumPrime(num)) return false;
    int digitSum = 0;
    while (num > 0) {
        digitSum += num % 10;
        num /= 10;
    }
    return isPrime(digitSum);
}

DEVICE int aurum_yotta_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isYottaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isBrontoAurumPrime(int num) {
    if (!isYottaAurumPrime(num)) return false;
    int digitProduct = 1;
    while (num > 0) {
        digitProduct *= num % 10;
        num /= 10;
    }
    return isPrime(digitProduct);
}

DEVICE int aurum_bronto_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isBrontoAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isGeoAurumPrime(int num) {
    if (!isBrontoAurumPrime(num)) return false;
    int digitSum = 0;
    while (num > 0) {
        digitSum += num % 10;
        num /= 10;
    }
    return isPrime(digitSum);
}

DEVICE int aurum_geo_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isGeoAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isAlphaAurumPrime(int num) {
    if (!isGeoAurumPrime(num)) return false;
    int digitProduct = 1;
    while (num > 0) {
        digitProduct *= num % 10;
        num /= 10;
    }
    return isPrime(digitProduct);
}

DEVICE int aurum_alpha_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isAlphaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isBetaAurumPrime(int num) {
    if (!isAlphaAurumPrime(num)) return false;
    int digitSum = 0;
    while (num > 0) {
        digitSum += num % 10;
        num /= 10;
    }
    return isPrime(digitSum);
}

DEVICE int aurum_beta_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isBetaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isGammaAurumPrime(int num) {
    if (!isBetaAurumPrime(num)) return false;
    int digitProduct = 1;
    while (num > 0) {
        digitProduct *= num % 10;
        num /= 10;
    }
    return isPrime(digitProduct);
}

DEVICE int aurum_gamma_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isGammaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isDeltaAurumPrime(int num) {
    if (!isGammaAurumPrime(num)) return false;
    int digitSum = 0;
    while (num > 0) {
        digitSum += num % 10;
        num /= 10;
    }
    return isPrime(digitSum);
}

DEVICE int aurum_delta_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isDeltaAurumPrime(i)) return i;
    }
    return -1;
}

HOST_DEVICE bool isEpsilonAurumPrime(int num) {
    if (!isDeltaAurumPrime(num)) return false;
    int digitProduct = 1;
    while (num > 0) {
        digitProduct *= num % 10;
        num /= 10;
    }
    return isPrime(digitProduct);
}

DEVICE int aurum_epsilon_prime(int start, int end) {
    for (int i = start; i < end; ++i) {
        if (isEpsilonAurumPrime(i)) return i;
    }
    return -1;
}
