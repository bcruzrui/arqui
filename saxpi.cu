#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <device_launch_parameters.h>


#define ARRAY_SIZE 1024*1024
#define NUM_THREADS 1024

// Saxpi 1 - Versión en C
void saxpi_c(int n, float a, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

__global__ void saxpi_1(int n, float a, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

__global__ void saxpi_1Block(int n, float a, float* x, float* y)
{
    int idx = threadIdx.x;
    int numElem = ARRAY_SIZE / NUM_THREADS;
    int offset = idx * numElem;
    if (offset + numElem < n) {
        for (int i = 0; i < numElem; i++)
            y[offset + i] = a * x[offset + i] + y[offset + i];
    }
}

__global__ void saxpi_nBlock(int n, float a, float* x, float* y)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}



int main(void)
{
    float* h_x, * h_y;  // Apuntadores en host
    float* d_x, * d_y; // Apuntadores en device
    cudaEvent_t inicioG, finG;
    cudaEventCreate(&inicioG);
    cudaEventCreate(&finG);

    size_t sz = ARRAY_SIZE * sizeof(float);

    clock_t inicio, fin;

    // Asigna memoria
    h_x = (float*)malloc(sz);
    h_y = (float*)malloc(sz);

    cudaMalloc(&d_x, sz);
    cudaMalloc(&d_y, sz);


    // inicializa arreglos
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    cudaMemcpy(d_x, h_x, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sz, cudaMemcpyHostToDevice);

    cudaEventRecord(inicioG);
    //saxpi_1Block <<<1, NUM_THREADS>>> (ARRAY_SIZE, 2.0, d_x, d_y);
    saxpi_nBlock <<< ARRAY_SIZE / NUM_THREADS, NUM_THREADS>>> (ARRAY_SIZE, 2.0, d_x, d_y);
    cudaEventRecord(finG);
    /*
    inicio = clock();
    saxpi_c(ARRAY_SIZE, 2.0, h_x, h_y);
    fin = clock();

    double t_exec = (double)(fin - inicio) / CLOCKS_PER_SEC;
    */

    cudaMemcpy(h_y, d_y, sz, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(finG);
    float t_exec = 0;
    cudaEventElapsedTime(&t_exec, inicioG, finG);

    printf("Algunos resultados: ");
    for (int i = 0; i < 10; i++) {
        printf("%3.2f, ", h_y[i]);
    }

    printf("\n Tiempo de ejecucion: %2.7f\n", t_exec);
    // Free memory
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}