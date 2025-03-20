#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

__global__ void factorboreljacobian1(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 2; i <= sqrt(n); ++i) {
            while (n % i == 0) {
                n /= i;
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian2(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 3; i <= sqrt(n); i += 2) {
            while (n % i == 0) {
                n /= i;
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian3(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 2; i <= sqrt(n); ++i) {
            while (n % i == 0) {
                n /= i;
            }
        }
        d_numbers[idx] = n + 1;
    }
}

__global__ void factorboreljacobian4(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 5; i <= sqrt(n); i += 6) {
            while (n % i == 0 || n % (i + 2) == 0) {
                n /= (n % i == 0 ? i : i + 2);
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian5(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 7; i <= sqrt(n); i += 10) {
            while (n % i == 0 || n % (i + 4) == 0 || n % (i + 6) == 0 || n % (i + 8) == 0) {
                n /= (n % i == 0 ? i : (n % (i + 4) == 0 ? i + 4 : (n % (i + 6) == 0 ? i + 6 : i + 8)));
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian6(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 11; i <= sqrt(n); i += 30) {
            while (n % i == 0 || n % (i + 2) == 0 || n % (i + 6) == 0 || n % (i + 8) == 0 ||
                   n % (i + 12) == 0 || n % (i + 14) == 0 || n % (i + 18) == 0 || n % (i + 20) == 0) {
                n /= (n % i == 0 ? i : (n % (i + 2) == 0 ? i + 2 : (n % (i + 6) == 0 ? i + 6 :
                    (n % (i + 8) == 0 ? i + 8 : (n % (i + 12) == 0 ? i + 12 : (n % (i + 14) == 0 ? i + 14 :
                        (n % (i + 18) == 0 ? i + 18 : i + 20))))));
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian7(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 13; i <= sqrt(n); i += 60) {
            while (n % i == 0 || n % (i + 2) == 0 || n % (i + 4) == 0 || n % (i + 8) == 0 ||
                   n % (i + 10) == 0 || n % (i + 12) == 0 || n % (i + 16) == 0 || n % (i + 18) == 0 ||
                   n % (i + 20) == 0 || n % (i + 22) == 0 || n % (i + 24) == 0 || n % (i + 26) == 0 ||
                   n % (i + 28) == 0 || n % (i + 30) == 0 || n % (i + 32) == 0 || n % (i + 34) == 0) {
                n /= (n % i == 0 ? i : (n % (i + 2) == 0 ? i + 2 : (n % (i + 4) == 0 ? i + 4 :
                    (n % (i + 8) == 0 ? i + 8 : (n % (i + 10) == 0 ? i + 10 : (n % (i + 12) == 0 ? i + 12 :
                        (n % (i + 16) == 0 ? i + 16 : (n % (i + 18) == 0 ? i + 18 : (n % (i + 20) == 0 ? i + 20 :
                            (n % (i + 22) == 0 ? i + 22 : (n % (i + 24) == 0 ? i + 24 : (n % (i + 26) == 0 ? i + 26 :
                                (n % (i + 28) == 0 ? i + 28 : (n % (i + 30) == 0 ? i + 30 : i + 32))))))))));
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian8(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 17; i <= sqrt(n); i += 210) {
            while (n % i == 0 || n % (i + 2) == 0 || n % (i + 4) == 0 || n % (i + 6) == 0 ||
                   n % (i + 8) == 0 || n % (i + 10) == 0 || n % (i + 12) == 0 || n % (i + 14) == 0 ||
                   n % (i + 16) == 0 || n % (i + 18) == 0 || n % (i + 20) == 0 || n % (i + 22) == 0 ||
                   n % (i + 24) == 0 || n % (i + 26) == 0 || n % (i + 28) == 0 || n % (i + 30) == 0 ||
                   n % (i + 32) == 0 || n % (i + 34) == 0 || n % (i + 36) == 0 || n % (i + 38) == 0 ||
                   n % (i + 40) == 0 || n % (i + 42) == 0 || n % (i + 44) == 0 || n % (i + 46) == 0 ||
                   n % (i + 48) == 0 || n % (i + 50) == 0 || n % (i + 52) == 0 || n % (i + 54) == 0 ||
                   n % (i + 56) == 0 || n % (i + 58) == 0 || n % (i + 60) == 0 || n % (i + 62) == 0 ||
                   n % (i + 64) == 0 || n % (i + 66) == 0 || n % (i + 68) == 0 || n % (i + 70) == 0) {
                n /= (n % i == 0 ? i : (n % (i + 2) == 0 ? i + 2 : (n % (i + 4) == 0 ? i + 4 :
                    (n % (i + 6) == 0 ? i + 6 : (n % (i + 8) == 0 ? i + 8 : (n % (i + 10) == 0 ? i + 10 : (n % (i + 12) == 0 ? i + 12 :
                        (n % (i + 14) == 0 ? i + 14 : (n % (i + 16) == 0 ? i + 16 : (n % (i + 18) == 0 ? i + 18 : (n % (i + 20) == 0 ? i + 20 :
                            (n % (i + 22) == 0 ? i + 22 : (n % (i + 24) == 0 ? i + 24 : (n % (i + 26) == 0 ? i + 26 :
                                (n % (i + 28) == 0 ? i + 28 : (n % (i + 30) == 0 ? i + 30 : (n % (i + 32) == 0 ? i + 32 :
                                    (n % (i + 34) == 0 ? i + 34 : (n % (i + 36) == 0 ? i + 36 : (n % (i + 38) == 0 ? i + 38 :
                                        (n % (i + 40) == 0 ? i + 40 : (n % (i + 42) == 0 ? i + 42 : (n % (i + 44) == 0 ? i + 44 :
                                            (n % (i + 46) == 0 ? i + 46 : (n % (i + 48) == 0 ? i + 48 : (n % (i + 50) == 0 ? i + 50 :
                                                (n % (i + 52) == 0 ? i + 52 : (n % (i + 54) == 0 ? i + 54 : (n % (i + 56) == 0 ? i + 56 :
                                                    (n % (i + 58) == 0 ? i + 58 : (n % (i + 60) == 0 ? i + 60 : (n % (i + 62) == 0 ? i + 62 :
                                                        (n % (i + 64) == 0 ? i + 64 : (n % (i + 66) == 0 ? i + 66 : (n % (i + 68) == 0 ? i + 68 : i + 70)))))))))))))))))))));
            }
        }
        d_numbers[idx] = n;
    }
}

__global__ void factorboreljacobian9(unsigned long long *d_numbers, unsigned int num_count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_count) {
        unsigned long long n = d_numbers[idx];
        for (unsigned long long i = 19; i <= sqrt(n); i += 2310) {
            while (n % i == 0 || n % (i + 2) == 0 || n % (i + 4) == 0 || n % (i + 6) == 0 || n % (i + 8) == 0 || n % (i + 10) == 0 ||
                   n % (i + 12) == 0 || n % (i + 14) == 0 || n % (i + 16) == 0 || n % (i + 18) == 0 || n % (i + 20) == 0 ||
                   n % (i + 22) == 0 || n % (i + 24) == 0 || n % (i + 26) == 0 || n % (i + 28) == 0 || n % (i + 30) == 0 ||
                   n % (i + 32) == 0 || n % (i + 34) == 0 || n % (i + 36) == 0 || n % (i + 38) == 0 || n % (i + 40) == 0 ||
                   n % (i + 42) == 0 || n % (i + 44) == 0 || n % (i + 46) == 0 || n % (i + 48) == 0 || n % (i + 50) == 0 ||
                   n % (i + 52) == 0 || n % (i + 54) == 0 || n % (i + 56) == 0 || n % (i + 58) == 0 || n % (i + 60) == 0 ||
                   n % (i + 62) == 0 || n % (i + 64) == 0 || n % (i + 66) == 0 || n % (i + 68) == 0 || n % (i + 70) == 0 ||
                   n % (i + 72) == 0 || n % (i + 74) == 0 || n % (i + 76) == 0 || n % (i + 78) == 0 || n % (i + 80) == 0 ||
                   n % (i + 82) == 0 || n % (i + 84) == 0 || n % (i + 86) == 0 || n % (i + 88) == 0 || n % (i + 90) == 0 ||
                   n % (i + 92) == 0 || n % (i + 94) == 0 || n % (i + 96) == 0 || n % (i + 98) == 0 || n % (i + 100) == 0) {
                n /= (n % i == 0 ? i : (n % (i + 2) == 0 ? i + 2 : (n % (i + 4) == 0 ? i + 4 :
                    (n % (i + 6) == 0 ? i + 6 : (n % (i + 8) == 0 ? i + 8 : (n % (i + 10) == 0 ? i + 10 : (n % (i + 12) == 0 ? i + 12 :
                        (n % (i + 14) == 0 ? i + 14 : (n % (i + 16) == 0 ? i + 16 : (n % (i + 18) == 0 ? i + 18 : (n % (i + 20) == 0 ? i + 20 :
                            (n % (i + 22) == 0 ? i + 22 : (n % (i + 24) == 0 ? i + 24 : (n % (i + 26) == 0 ? i + 26 :
                                (n % (i + 28) == 0 ? i + 28 : (n % (i + 30) == 0 ? i + 30 : (n % (i + 32) == 0 ? i + 32 :
                                    (n % (i + 34) == 0 ? i + 34 : (n % (i + 36) == 0 ? i + 36 : (n % (i + 38) == 0 ? i + 38 : (n % (i + 40) == 0 ? i + 40 :
                                        (n % (i + 42) == 0 ? i + 42 : (n % (i + 44) == 0 ? i + 44 : (n % (i + 46) == 0 ? i + 46 : (n % (i + 48) == 0 ? i + 48 :
                                            (n % (i + 50) == 0 ? i + 50 : (n % (i + 52) == 0 ? i + 52 : (n % (i + 54) == 0 ? i + 54 : (n % (i + 56) == 0 ? i + 56 :
                                                (n % (i + 58) == 0 ? i + 58 : (n % (i + 60) == 0 ? i + 60 : (n % (i + 62) == 0 ? i + 62 : (n % (i + 64) == 0 ? i + 64 :
                                                    (n % (i + 66) == 0 ? i + 66 : (n % (i + 68) == 0 ? i + 68 : (n % (i + 70) == 0 ? i + 70 :
                                                        (n % (i + 72) == 0 ? i + 72 : (n % (i + 74) == 0 ? i + 74 : (n % (i + 76) == 0 ? i + 76 : (n % (i + 78) == 0 ? i + 78 :
                                                            (n % (i + 80) == 0 ? i + 80 : (n % (i + 82) == 0 ? i + 82 : (n % (i + 84) == 0 ? i + 84 : (n % (i + 86) == 0 ? i + 86 :
                                                                (n % (i + 88) == 0 ? i + 88 : (n % (i + 90) == 0 ? i + 90 : (n % (i + 92) == 0 ? i + 92 : (n % (i + 94) == 0 ? i + 94 :
                                                                    (n % (i + 96) == 0 ? i + 96 : (n % (i + 98) == 0 ? i + 98 : (n % (i + 100) == 0 ? i + 100 : -1))))))))))))))))))))))))))))))))))))))))));
            }
            d[i] = n;
        }
    }

    for(int i=0;i<N;i++){
        printf("%d ", d[i]);
    }

    free(d);
    return 0;
}
