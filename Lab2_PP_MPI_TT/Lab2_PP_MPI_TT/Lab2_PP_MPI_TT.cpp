#include "mpi.h"
#include <iostream>

#define Q 21
#define N_SIZE 5600
#define NUM_VEC 4
#define ADDITION 33//33461
#define ITERS_NUMBER 20

double sequental_time(double* a, double* b, double* c, double* d, double* s) {
    double start_time, total_time = 0, total_seq_sum = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = MPI_Wtime();

        for (i = 0; i < N_SIZE; ++i) {
            for (j = 0; j < Q; ++j) {
                s[i] = a[i] + b[i] + c[i] + d[i];
                total_seq_sum += s[i];
            }
        }
        total_time = total_time + (MPI_Wtime() - start_time);

    }
    total_seq_sum /= (Q * ITERS_NUMBER);
    std::cout << std::endl << "Sequential total_sum: " << total_seq_sum <<  std::endl;
    std::cout << "first and last 5 elements of sum: ";
    for (i = 0; i < N_SIZE; ++i) {
        if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
            std::cout << s[i] << " ";
        }
    }
    std::cout << std::endl;
    return total_time / ITERS_NUMBER;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int ProcRank, ProcNum;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    int i;
    double seq_time = 0, start_time = 0;

    double* a = NULL, *b = NULL, *c = NULL, *d = NULL, *s = NULL;

    if (ProcRank == 0) {
        std::cout << "Type: double" << std::endl << "Number of vectors: " << NUM_VEC << std::endl;
        std::cout << "Vector size: " << N_SIZE << std::endl << "Number od threads: [4, 8, 16]" << std::endl;
        std::cout << "Parameter Q = " << Q << std::endl;

        a = new double[N_SIZE];
        b = new double[N_SIZE];
        c = new double[N_SIZE];
        d = new double[N_SIZE];
        s = new double[N_SIZE];

        for (i = 0; i < N_SIZE; ++i) {
            a[i] = 0.1;
            b[i] = 0.1;
            c[i] = 0.1;
            d[i] = 0.1;
        }
        seq_time = sequental_time(a, b, c, d, s);
    }

    //количетво элементов кратно количеству потоков
    int count_of_elem = N_SIZE / ProcNum; 

    double time_Scatter = 0, time_Scatterv = 0, time_mul = 0, time_non = 0;
    double total_sum_mul = 0, total_sum_non = 0;

    double* a_loc = new double[count_of_elem];
    double* b_loc = new double[count_of_elem];
    double* c_loc = new double[count_of_elem];
    double* d_loc = new double[count_of_elem];
    double* s_loc = new double[count_of_elem];
    double* sum_scat = new double[N_SIZE];

    for (int count = 0; count < ITERS_NUMBER; ++count) {
        start_time = MPI_Wtime();

        MPI_Scatter(a, count_of_elem, MPI_DOUBLE, a_loc, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, count_of_elem, MPI_DOUBLE, b_loc, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(c, count_of_elem, MPI_DOUBLE, c_loc, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(d, count_of_elem, MPI_DOUBLE, d_loc, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        time_Scatter += MPI_Wtime() - start_time;

        for (int i = 0; i < count_of_elem; ++i) {
            for (int q = 0; q < Q; ++q) {
                s_loc[i] = a_loc[i] + b_loc[i] + c_loc[i] + d_loc[i];
            }
        }
        MPI_Gather(s_loc, count_of_elem, MPI_DOUBLE, sum_scat, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        time_mul += MPI_Wtime() - start_time;

        if (ProcRank == 0) {
            for (int i = 0; i < N_SIZE; ++i)
                total_sum_mul += sum_scat[i];
        }
    }

    total_sum_mul /= ITERS_NUMBER;

    time_mul /= ITERS_NUMBER;
    time_Scatter /= ITERS_NUMBER;

    MPI_Barrier(MPI_COMM_WORLD);

    if (ProcRank == 0) {
        delete[] a;
        delete[] b;
        delete[] c;
        delete[] d;
        delete[] s;
    }

    // Некратный вариант (non)   

    count_of_elem = (N_SIZE + ADDITION) / ProcNum;

    if (ProcRank == 0) {
        a = new double[N_SIZE + ADDITION];
        b = new double[N_SIZE + ADDITION];
        c = new double[N_SIZE + ADDITION];
        d = new double[N_SIZE + ADDITION];
        s = new double[N_SIZE + ADDITION];

        for (i = 0; i < N_SIZE + ADDITION; ++i) {
            a[i] = 0.1;
            b[i] = 0.1;
            c[i] = 0.1;
            d[i] = 0.1;
        }
    }
    double* a_loc_V = new double[count_of_elem];
    double* b_loc_V = new double[count_of_elem];
    double* c_loc_V = new double[count_of_elem];
    double* d_loc_V = new double[count_of_elem];
    double* s_loc_V = new double[count_of_elem];

    double* sum_V = new double[N_SIZE + ADDITION];

    int* displs = new int[ProcNum];
    displs[0] = 0;
    int* counts = new int[ProcNum]; 
    int extra = (N_SIZE + ADDITION) % ProcNum;// остаток

    for (int count = 0; count < ITERS_NUMBER; ++count) {
        start_time = MPI_Wtime();

        for (int i = 0; i < ProcNum; ++i) {
            counts[i] = (N_SIZE + ADDITION) / ProcNum;

            if (i < extra) counts[i] += 1;
            if (i == 0) continue;

            displs[i] = displs[i - 1] + counts[i - 1];
        }

        count_of_elem = counts[ProcRank];

        MPI_Scatterv(a, counts, displs, MPI_DOUBLE, a_loc_V, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(b, counts, displs, MPI_DOUBLE, b_loc_V, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(c, counts, displs, MPI_DOUBLE, c_loc_V, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(d, counts, displs, MPI_DOUBLE, d_loc_V, count_of_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        time_Scatterv += MPI_Wtime() - start_time;

        for (int i = 0; i < count_of_elem; ++i) {
            for (int q = 0; q < Q; ++q) {
                s_loc_V[i] = a_loc_V[i] + b_loc_V[i] + c_loc_V[i] + d_loc_V[i];
            }
        }
        MPI_Gatherv(s_loc_V, counts[ProcRank], MPI_DOUBLE, sum_V, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        time_non += MPI_Wtime() - start_time;

        if (ProcRank == 0) {
            for (int i = 0; i < N_SIZE + ADDITION; ++i)
                total_sum_non += sum_V[i];
        }
    }
    total_sum_non /= ITERS_NUMBER;

    time_Scatterv /= ITERS_NUMBER;
    time_non /= ITERS_NUMBER;

    if (ProcRank == 0) {

        std::cout << "Sequential time: " << seq_time << std::endl;

        std::cout << std::endl << "Multiple total_sum: " << total_sum_mul << std::endl;
        std::cout << "Multiple time: " << time_mul << std::endl;
        std::cout << "Multiple acceleration with Scatter: " << seq_time / (time_mul + time_Scatter) << std::endl;
        std::cout << "Multiple acceleration without: " << seq_time / time_mul << std::endl;

        std::cout << "first and last 5 elements of sum: ";
        for (i = 0; i < N_SIZE; ++i) {
            if ((i >= 0 and i < 5) or (i < N_SIZE and i >= N_SIZE - 5)) {
                 std::cout << sum_scat[i] << " ";
            }
        }
        std::cout << std::endl;

        std::cout << std::endl << "Non-multiple total_sum: " << total_sum_non << std::endl;
        std::cout << "Non-multiple time: " << time_non << std::endl;
        std::cout << "Non-multiple acceleration with ScatterV: " << seq_time / (time_non + time_Scatterv)<< std::endl;
        std::cout << "Non-multiple acceleration without: " << seq_time / time_non << std::endl;

        std::cout << "first and last 5 elements of sum: ";
        for (i = 0; i < N_SIZE + ADDITION; ++i) {
            if ((i >= 0 and i < 5) or (i < N_SIZE + ADDITION and i >= N_SIZE + ADDITION - 5)) {
                std::cout << sum_V[i] << " ";
            }
        }
        std::cout << std::endl;

        std::cout << std::endl << "Scatter time: " << time_Scatter << " sec\n";
        std::cout << "Scatterv time: " << time_Scatterv << " sec\n";
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] s;

    delete[] a_loc;
    delete[] b_loc;
    delete[] c_loc;
    delete[] d_loc;
    delete[] s_loc;

    delete[] a_loc_V;
    delete[] b_loc_V;
    delete[] c_loc_V;
    delete[] d_loc_V;
    delete[] s_loc_V;

    delete[] sum_V;
    delete[] sum_scat;
    MPI_Finalize();
    return 0;
}