/*
 *
 * Programa de Introducci�n a los conceptos de CUDA
 * Suma dos vectores de enteros e indica qu� partes del c�digo deben modificarse
 * para implementar la versi�n paralela en el GPU
 *
 * Asume un modelo de memoria distribuida
 */

 /* Parte 0: A�adir los archivos .h de CUDA*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>


/* Numero de elementos en el vector */
#define ARRAY_SIZE 256 * 1024

/*
 * N�mero de bloques e hilos
 * Su producto siempre debe ser el tama�o del vector (arreglo).
 */
#define NUM_BLOCKS  256
#define THREADS_PER_BLOCK 1024


 /* Declaraci�n de m�todos/


 /* Utilidad para checar errores de CUDA */
 void checkCUDAError(const char*);

 /* Funci�n en C para suma de vectores*/
void vect_add_c(int* a, int* b, int* c) {
    printf("Ejecuci�n secuencial \n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        c[i] = a[i] + b[i];
    }
}
/* Kernel para sumar dos vectores en un s�lo bloque de hilos */
__global__ void vect_add(int* d_a, int* d_b, int* d_c)
{
    /* Part 2B: Implementaci�n del kernel para realizar la suma de los vectores en el GPU */
    int idx = threadIdx.x;
    d_c[idx] = d_a[idx] + d_b[idx];
}

/* Versi�n de m�ltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int* d_a, int* d_b, int* d_c)
{
    /* Part 2C: Implementaci�n del kernel pero esta vez permitiendo m�ltiples bloques de hilos. */
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < ARRAY_SIZE) { 
        d_c[idx] = d_a[idx] + d_b[idx]; 
    }
}

 /* Main routine */
int main(int argc, char* argv[])
{
    int* h_a, * h_b, * h_c; /* Arreglos del CPU */
    int* d_a, * d_b, * d_c;/* Arreglos del GPU */

//    cudaError_t err = cudaSuccess;  // Para checar errores en CUDA

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    h_a = (int*)malloc(sz);
    h_b = (int*)malloc(sz);
    h_c = (int*)malloc(sz);

    /*
    * Parte 1A:Reservar memoria en el GPU
    */
    cudaMalloc(&d_a, sz);
    cudaMalloc(&d_b, sz);
    cudaMalloc(&d_c, sz);

    /* inicializaci�n */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_b[i] = i + 10;
        h_c[i] = 0;
    }


    /* Parte 1B: Copiar los vectores del CPU al GPU */
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz, cudaMemcpyHostToDevice);


    /* Parte 2A: Configurar y llamar los kernels o la funci�n */
    dim3 dimGrid(NUM_BLOCKS);
    dim3 dimBlock(THREADS_PER_BLOCK);
    const clock_t begin_time = clock();
    vect_add_multiblock<<<dimGrid,dimBlock>>>(d_a, d_b, d_c);
    //vect_add_c(h_a, h_b, h_c);
    printf("Tiempo de ejecuci�n: %f \n", float(clock() - begin_time) / CLOCKS_PER_SEC);

    /* Esperar a que todos los threads acaben y checar por errores */

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "No se pudo lanzar el kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    printf("Algunos resultados: ");
    for (i = 0; i < 10; i++) {
        printf("%d, ", h_c[i]);
    }
    printf("\n\n");
    for (i = ARRAY_SIZE - 10; i < ARRAY_SIZE; i++) {
        printf("%d, ", h_c[i]);
    }
    printf("\n\n");

    /* Parte 1D: Liberar los arreglos */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}