#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define Q 21
#define N_SIZE 5600000
#define NUM_VEC 4
#define CHUNK 100 

int main() {
    std::cout << "Type: double" << std::endl << "Number of vectors: " << NUM_VEC << std::endl;
    std::cout << "Vector size: " << N_SIZE << std::endl << "Number od threads: [4, 8, 16]" << std::endl;
    std::cout << "Parameter Q = " << Q << std::endl;

    double* a = new double[N_SIZE];
    double* b = new double[N_SIZE];
    double* c = new double[N_SIZE];
    double* d = new double[N_SIZE];
    double* sum = new double[N_SIZE];


    // Инициализация массивов
    for (int i = 0; i < N_SIZE; ++i) {
        a[i] = 0.1;
        b[i] = 0.1;
        c[i] = 0.1;
        d[i] = 0.1;
    }

    int num_iterations = 20, i, q, chunk;
    chunk = CHUNK;
    double start_time, end_time, time_st = 0, time_dy = 0, time_gu = 0, time_seq = 0;

    omp_set_num_threads(4);

    // Последовательное суммирование
    start_time = omp_get_wtime();
    for (int count = 0; count < num_iterations; count++) {
        for (int i = 0; i < N_SIZE; i++) {
            for (int j = 0; j < Q; ++j) {
                sum[i] = a[i] + b[i] + c[i] + d[i];
            }
        }
    }
    time_seq = (omp_get_wtime() - start_time) / num_iterations;
    // Вывод первых и последних пяти элементов
    std::cout << std::endl << "Sequential time: " << time_seq << std::endl;
    std::cout << "first and last 5 elements of sum: ";
    for (i = 0; i < N_SIZE; ++i) {
        if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
            std::cout << sum[i] << " ";
        }
    }
    std::cout << std::endl;

    double total_sum = 0;
    for (i = 0; i < N_SIZE; ++i) {
        total_sum += sum[i];
    }
    std::cout << "Sequential total_sum: " << total_sum << std::endl;

    // инициализация
    double parTime = 0;
    for (int count = 0; count < num_iterations; count++) {
        start_time = omp_get_wtime();
#pragma omp parallel
        {}
        end_time = omp_get_wtime();
        parTime += (end_time - start_time); // Накапливаем общее время
    }
    parTime /= num_iterations;
    std::cout << std::endl << "Parallel time: " << parTime << std::endl;

    //суммирование векторов
    //static
    for (int count = 0; count < num_iterations; count++) {
        start_time = omp_get_wtime();
#pragma omp parallel for shared(a, b, c, d, sum) private(i, q) schedule(static, CHUNK)
        for (i = 0; i < N_SIZE; ++i) {
            for (q = 0; q < Q; ++q) {
                sum[i] = a[i] + b[i] + c[i] + d[i];
            }
        }
        time_st += (omp_get_wtime() - start_time);
    }
    time_st /= num_iterations;
    std::cout  << std::endl<< "Static time: " << time_st << std::endl;
    total_sum = 0;

    std::cout << "first and last 5 elements of sum: ";
    for (i = 0; i < N_SIZE; ++i) {
        if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
            std::cout << sum[i] << " ";
        }
    }
    std::cout << std::endl;

    for (i = 0; i < N_SIZE; ++i) {
        total_sum += sum[i];
    }
    std::cout << "Static total_sum: " << total_sum << std::endl;
    std::cout << "Static acceleration with: " << time_seq / time_st << std::endl;
    std::cout << "Static acceleration without: " << time_seq / (time_st - parTime) << std::endl;

    //dynamic
    for (int count = 0; count < num_iterations; count++) {
        start_time = omp_get_wtime();
#pragma omp parallel for shared(a, b, sum) private(i, q) schedule(dynamic, CHUNK)
        for (i = 0; i < N_SIZE; ++i) {
            for (q = 0; q < Q; ++q) {
                sum[i] = a[i] + b[i] + c[i] + d[i];
            }
        }
        time_dy += (omp_get_wtime() - start_time);
    }
    time_dy /= num_iterations;
    std::cout << std::endl << "Dynamic time: " << time_dy << std::endl;
    total_sum = 0;

    std::cout << "first and last 5 elements of sum: ";
    for (i = 0; i < N_SIZE; ++i) {
        if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
            std::cout << sum[i] << " ";
        }
    }
    std::cout << std::endl;

    for (i = 0; i < N_SIZE; ++i) {
        total_sum += sum[i];
    }
    std::cout << "Dynamic total_sum: " << total_sum<< std::endl;
    std::cout << "Dynamic acceleration with: " << time_seq / time_dy << std::endl;
    std::cout << "Dynamic acceleration without: " << time_seq / (time_dy - parTime) << std::endl;

    // guided
    for (int count = 0; count < num_iterations; count++) {
        start_time = omp_get_wtime();
#pragma omp parallel for shared(a, b, sum) private(i, q) schedule(guided, CHUNK)
        for (i = 0; i < N_SIZE; ++i) {
            for (q = 0; q < Q; ++q) {
                sum[i] = a[i] + b[i] + c[i] + d[i];
            }
        }
        time_gu += (omp_get_wtime() - start_time);
    }
    time_gu /= num_iterations;
    std::cout << std::endl << "Guided time: " << time_gu << std::endl;

    std::cout << "first and last 5 elements of sum: ";
    for (i = 0; i < N_SIZE; ++i) {
        if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
            std::cout << sum[i] << " ";
        }
    }
    std::cout << std::endl;

    total_sum = 0;
    for (i = 0; i < N_SIZE; ++i) {
        total_sum += sum[i];
    }
    std::cout << "Guided total_sum: " << total_sum << std::endl;
    std::cout << "Guided acceleration with: " << time_seq / time_gu << std::endl;
    std::cout << "Guided acceleration without: " << time_seq / (time_gu - parTime) << std::endl;

    // Освобождение памяти
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] sum;


    return 0;
}