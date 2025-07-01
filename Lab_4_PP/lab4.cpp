#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

// [(128,1024);(256,1024);(256,512);(128,512);(256,256);(128,256)]

#define NMAX 5600000
#define ITERATIONS 20

__global__ void addKernel(double* sum, double* a, double* b, double* c, double* d, unsigned int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        sum[i] = a[i] + b[i] + c[i] + d[i];
}

int main(int argc, char* argv[])
{
    int GRID_SIZE;
    int BLOCK_DIM;

    int n2b = NMAX * sizeof(double);
    float time_seq = 0.0, time_tr = 0.0;
    float time_gpu = 0.0;

    if (argc > 2) {
        GRID_SIZE = atoi(argv[1]);
        BLOCK_DIM = atoi(argv[2]);
    }

    double* a = (double*)calloc(NMAX, sizeof(double));
    double* b = (double*)calloc(NMAX, sizeof(double));
    double* c = (double*)calloc(NMAX, sizeof(double));
    double* d = (double*)calloc(NMAX, sizeof(double));
    double* sum_seq = (double*)calloc(NMAX, sizeof(double));
    double* sum_cuda = (double*)calloc(NMAX, sizeof(double));

    for (int i = 0; i < NMAX; i++) {
        a[i] = 1.0;
        b[i] = 1.0;
        c[i] = 1.0;
        d[i] = 1.0;
        sum_seq[i] = 0.0;
        sum_cuda[i] = 0.0;
    }

    double* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    double* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    double* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    double* ddev = NULL;
    cuerr = cudaMalloc((void**)&ddev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for d: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    double* sumdev = NULL;
    cuerr = cudaMalloc((void**)&sumdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for sum: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cudaEvent_t start, stop;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA stop event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    for (int j = 0; j < ITERATIONS; j++) {

        float seqTime_tmp = 0.0;
        // последовательный алгоритм
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot create CUDA start event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        for (int i = 0; i < NMAX; i++) {
            sum_seq[i] = a[i] + b[i] + c[i] + d[i];
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot create CUDA start event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&seqTime_tmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot calculate elapsed time: %s\n", cudaGetErrorString(cuerr));
            return 0;
        }



        //«амер времени передачи данных на видеокарту
        float tr1_tmp = 0.0;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot create CUDA start event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot copy a array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        cuerr = cudaMemcpy(bdev, b, n2b, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot copy b array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        cuerr = cudaMemcpy(cdev, c, n2b, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot copy a array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        cuerr = cudaMemcpy(ddev, d, n2b, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot copy b array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record CUDA stop event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&tr1_tmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot calculate elapsed time: %s\n", cudaGetErrorString(cuerr));
            return 0;
        }




        // €дро
        float gpuTime_tmp = 0.0;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record start CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        addKernel << < GRID_SIZE, BLOCK_DIM >> > (sumdev, adev, bdev, cdev, ddev, NMAX);

        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record stop CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&gpuTime_tmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot calculate elapsed time: %s\n", cudaGetErrorString(cuerr));
            return 0;
        }

        float tr2_time_tmp = 0.0;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record start CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaMemcpy(sum_cuda, sumdev, n2b, cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot copy c array from device to host: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record stop CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&tr2_time_tmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot calculate elapsed time: %s\n", cudaGetErrorString(cuerr));
            return 0;
        }


        time_seq += seqTime_tmp;
        time_gpu += gpuTime_tmp;
        time_tr += tr1_tmp;
        time_tr += tr2_time_tmp;
    }

    printf("Number of elements = %d\n", NMAX);
    printf("Number of vectors: 4\n");
    printf("GRID_DIM = %d, BLOCK_DIM = %d\n\n", GRID_SIZE, BLOCK_DIM);

    printf("sequential time: %f milliseconds\n", time_seq / ITERATIONS);
    printf("time tr %.9f milliseconds\n", time_tr / ITERATIONS);
    printf("time kernel %.9f milliseconds\n\n", time_gpu / ITERATIONS);

    printf("First 3 elements\n");
    for (int i = 0; i < 3; i++) {
        printf("Element %d: sum_seq: %.2f sum_cuda: %.2f \n", i, sum_seq[i], sum_cuda[i]);
    }
    printf("Last 3 elements\n");
    for (int i = -3; i < 0; i++) {
        printf("Element n%d: sum_seq: %.2f sum_cuda: %.2f \n", i, sum_seq[NMAX + i], sum_cuda[NMAX + i]);
    }
    printf("\n");
    printf("Acceleration with send: %f\n", time_seq / (time_gpu + time_tr));
    printf("Acceleration without send: %f\n", time_seq / time_gpu);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    cudaFree(ddev);
    cudaFree(sumdev);
    free(a);
    free(b);
    free(c);
    free(d);
    free(sum_seq);
    free(sum_cuda);

    return 0;
}