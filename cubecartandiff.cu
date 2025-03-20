#ifndef CUBECARTANDIFF_H
#define CUBECARTANDIFF_H

__device__ bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

__device__ int next_prime(int n) {
    while (!is_prime(n)) ++n;
    return n;
}

__device__ bool is_cubic_number(int n) {
    int cube_root = round(pow(n, 1.0 / 3.0));
    return cube_root * cube_root * cube_root == n;
}

__device__ int next_cubic_prime(int n) {
    while (!is_prime(n) || !is_cubic_number(n)) ++n;
    return n;
}

__device__ bool is_cartesian_coordinate_prime(int x, int y, int z) {
    return is_prime(x) && is_prime(y) && is_prime(z);
}

__device__ int diff_of_primes(int a, int b) {
    return abs(next_prime(a) - next_prime(b));
}

__device__ int sum_of_cubic_primes(int n) {
    int sum = 0;
    for (int i = 2; i <= n; ++i)
        if (is_prime(i) && is_cubic_number(i))
            sum += i;
    return sum;
}

__device__ bool is_sum_of_squares_prime(int a, int b) {
    return is_prime(a * a + b * b);
}

__device__ int largest_prime_below_n(int n) {
    while (!is_prime(n)) --n;
    return n;
}

__device__ bool are_consecutive_primes(int a, int b) {
    return next_prime(a) == b;
}

__device__ int smallest_prime_above_n(int n) {
    while (!is_prime(n)) ++n;
    return n;
}

__device__ bool is_product_of_primes(int n) {
    for (int i = 2; i * i <= n; ++i)
        if (is_prime(i) && n % i == 0)
            return true;
    return false;
}

__device__ int count_primes_below_n(int n) {
    int count = 0;
    for (int i = 2; i < n; ++i)
        if (is_prime(i)) ++count;
    return count;
}

__device__ bool is_difference_of_cubes_prime(int a, int b) {
    return is_prime(abs(a * a * a - b * b * b));
}

__device__ int sum_of_primes_in_range(int start, int end) {
    int sum = 0;
    for (int i = start; i <= end; ++i)
        if (is_prime(i)) sum += i;
    return sum;
}

__device__ bool is_quartic_number(int n) {
    int quartic_root = round(pow(n, 1.0 / 4.0));
    return quartic_root * quartic_root * quartic_root * quartic_root == n;
}

__device__ int largest_cubic_prime_below_n(int n) {
    while (!is_prime(n) || !is_cubic_number(n)) --n;
    return n;
}

__device__ bool is_product_of_consecutive_primes(int a, int b) {
    return are_consecutive_primes(a, b) && is_prime(a * b);
}

__device__ int smallest_cubic_prime_above_n(int n) {
    while (!is_prime(n) || !is_cubic_number(n)) ++n;
    return n;
}

__device__ bool is_sum_of_four_squares_prime(int a, int b, int c, int d) {
    return is_prime(a * a + b * b + c * c + d * d);
}

__device__ int sum_of_primes_divisible_by_k(int n, int k) {
    int sum = 0;
    for (int i = 2; i <= n; ++i)
        if (is_prime(i) && i % k == 0) sum += i;
    return sum;
}
