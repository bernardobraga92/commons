#include <cuda_runtime.h>
#include <iostream>

__global__ void PerfectGodelComplexConj_Func1(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func2(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = nextPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func3(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = prevPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func4(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = primeFactorization(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func5(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isMersennePrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func6(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isFermatPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func7(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isSophieGermainPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func8(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isCarmichaelNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func9(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isEulerPseudoprime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func10(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isStrongPseudoprime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func11(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isLucasPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func12(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isWieferichPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func13(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isWilsonPrime(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func14(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isSierpinskiNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func15(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isRieselNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func16(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isFriendlyNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func17(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isAliquotSequence(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func18(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isPerfectNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func19(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isAbundantNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func20(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isDeficientNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func21(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isHarshadNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func22(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isNarcissisticNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func23(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isArmstrongNumber(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func24(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isPluperfectDigitalInvariant(data[i]);
    }
}

__global__ void PerfectGodelComplexConj_Func25(unsigned long long *data, unsigned int size) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] = isLychrelNumber(data[i]);
    }
}
