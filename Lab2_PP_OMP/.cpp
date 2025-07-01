#include <iostream>
#include <time.h>
#define NMAX 5600000
#define NUM_VEC 4
#define Q 21


int main(int argc, char* argv[]) {
    const int iterations = 20;
    long double sum = 0.0;
    double total_time = 0;

    // Создание массивов
    double a[NMAX], b[NMAX], c[NMAX], d[NMAX];

    for (int i = 0; i < NMAX; ++i) {
        a[i] = 0.11;
        b[i] = 1.11;
        c[i] = 0.21;
        d[i] = 1.31;
    }

    clock_t start_time, end_time;

    sum = 0;
    // Последовательное суммирование
    for (int count = 0; count < iterations; ++count) {
        start_time = clock();
        for (int i = 0; i < NMAX; i++) {
            for (int j = 0; j < Q; ++j) {
                sum += a[i] + b[i] + c[i] + d[i];
            }
        }
        end_time = clock();
        end_time = end_time - start_time;
        total_time += end_time;
    }

    sum = sum / (iterations * Q);
    std::cout << "\nConsistent Q: %d" << Q << std::endl << "\nTotal Sum = %10.2f" << sum << std::endl;
    std::cout << "\nTIME OF WORK IS %f " << total_time / CLOCKS_PER_SEC / iterations << std::endl;

    sum = 0;
    total_time = 0;

    return 0;
}