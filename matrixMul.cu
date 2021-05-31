#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <device_launch_parameters.h>


#define NUM_THREADS 32
#define BLOCK_SIZE 32


__global__ void multiplication_nBlock(float *a, float* b, float* c)
{
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    float sum = 0;
    //printf("row: %d, col: %d \n", row, col);
    if (row < NUM_THREADS && col < NUM_THREADS) {

        for (int i = 0; i < NUM_THREADS; i++) {
            sum += a[row * NUM_THREADS + i] * b[i * NUM_THREADS + col];
        }
    }
    
    c[row * NUM_THREADS + col] = sum;
}

__global__ void multiplication_nBlock_shared(float* a, float* b, float* c)
{

    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    float sum = 0;
    __shared__ float  a[NUM_THREADS][NUM_THREADS], b[NUM_THREADS][NUM_THREADS]; 
    if (row < NUM_THREADS && col < NUM_THREADS) {
        for (int i = 0; i < NUM_THREADS; i++) {
            sum += a[row * NUM_THREADS + i] * b[i * NUM_THREADS + col];
        }
        __syncthreads();
    }
    c[row * NUM_THREADS + col] = sum;
}

int main(void)
{
    float* h_a, * h_b, * h_c;  // Apuntadores en host
    float* d_a, * d_b, * d_c; // Apuntadores en device
    cudaEvent_t inicioG, finG;
    cudaEventCreate(&inicioG);
    cudaEventCreate(&finG);

    int size = NUM_THREADS * NUM_THREADS;
    size_t sz = size * sizeof(float);

    // Asigna memoria
    h_a = (float*)malloc(sz);
    h_b = (float*)malloc(sz);
    h_c = (float*)malloc(sz);

    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);


    // inicializa arreglos
    for (int i = 0; i < size; i++) {
        h_a[i] = 3.0f;
        h_b[i] = 4.0f;
        h_c[i] = 0.0;
    }

    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz, cudaMemcpyHostToDevice);

    dim3 dimGrid(NUM_THREADS, NUM_THREADS);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cudaEventRecord(inicioG);
    multiplication_nBlock<<<dimBlock, dimGrid>>>(d_a, d_b, d_c);
    cudaEventRecord(finG);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "No se pudo lanzar el kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
   

    cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(finG);
    float t_exec = 0;
    cudaEventElapsedTime(&t_exec, inicioG, finG);

    printf("Algunos resultados: ");
    for (int i = 0; i < 100; i++) {
        printf("%3.2f, ", h_c[i]);
    }

    printf("\n\n");
    for (int i = size - 100; i < size; i++) {
        printf("%3.2f, ", h_c[i]);
    }

    printf("\n Tiempo de ejecucion: %2.7f\n", t_exec);
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}