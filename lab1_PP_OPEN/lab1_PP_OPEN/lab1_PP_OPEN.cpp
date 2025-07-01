#include <iostream>
#include <omp.h>
#include <ctime>

int main() {
    // Указываем количество потоков
    omp_set_num_threads(12);
    std::cout << "Count of threads: " << omp_get_max_threads() << std::endl;

    // Указываем параллельную часть кода
#pragma omp parallel
    {
        double start_time = omp_get_wtime();
        int thread = omp_get_thread_num();

        //только один поток в данный момент времени выполняет вывод:
#pragma omp critical
        {
            double end_time = omp_get_wtime();
            std::cout << "It's thread number " << thread << ", lead time: " << end_time - start_time << " seconds." << std::endl;
        }
    }

    return 0;
}
