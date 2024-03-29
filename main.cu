#include <stdio.h>
#include <stdlib.h>
#include "gsw.c"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}


#define THREADLIMITPERBLOCK 512 

__global__ void PrintMatrix(int * m1, int total) {
    int id = threadIdx.x + blockIdx.x * blockDim.x; 
    if(id < total){
        printf(" %d", m1[id]);
    }
}

__global__ void MatMultiplication(int * lm1, int * cm2, int * rm3, int total, int width) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ri = id % width;
    int ci = (id / width);
    if(id < total){
        int sum = 0;
        for(int i = 0; i < width; i++){
            sum += lm1[ri*width + i] * cm2[ci*width + i];
        }
        rm3[ri*width + ci] = sum;
    }
}

__global__ void MatSum(int * lm1, int * cm2, int * rm3, int total, int width) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ri = id % width;
    int ci = (id / width);
    if(id < total){
        rm3[ri*width + ci] = (lm1[ri*width + ci] + cm2[ri*width + ci]);
    }
}

__global__ void MatSub(int * lm1, int * cm2, int * rm3, int total, int width) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ri = id % width;
    int ci = (id / width);
    if(id < total){
        rm3[ri*width + ci] = (lm1[ri*width + ci] - cm2[ri*width + ci]);
    }
}

__global__ void GPUBitDecompInverse(int * bv, int * rv, int total, int l) { // total must not be the number of bits, but the number of elements in rv
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < total){
        rv[id] = 0;
        for(int i = 0; i < l; i++){
            rv[id] += (1 << (i)) * bv[id*l + i];
        }
    }
}

__global__ void GPUBitDecomp(int * v, int * rv, int total, int l) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < total){
        //printf(" [%d %d %d]", v[id], id*l, id*l + 3);
        for(int i = 0; i < l; i++){
            rv[id*l + i] = (v[id] >> i) & 1;
            printf("[%d %d] ", id*l + i, rv[id*l + i]);
        }
    }
}

dim3 getDim3ForMatrix(int rows, int columns){
    int numberOfThreads = rows * columns;
    int requiredNumberOfThreadsPerBlock = 1;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if(numberOfThreads > THREADLIMITPERBLOCK) {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = ceil (numberOfThreads / pw)  ;
    }

    dim3 grid(requiredNumberOfThreadsPerBlock, 1, 1);
    dim3 block(pw, 1, 1);

    return block;
}

int *  MatrixAllocOnDevice(int ** matrix, int rows, int columns, cudaStream_t st){
    int *deviceMatrix;
    cudaMallocAsync((void**)&deviceMatrix,  rows * columns *sizeof(int), st);
    for(int i=0; i<rows; i++) {
        cudaMemcpyAsync (&deviceMatrix[i*rows], matrix[i], (columns)*sizeof(int), cudaMemcpyHostToDevice, st);
    }
    return deviceMatrix;
}

int **  MatrixAllocOnHost(int * deviceMatrix, int rows, int columns, cudaStream_t st){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for(int i=0; i<rows; i++) {
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        cudaMemcpyAsync (matrix[i], &deviceMatrix[i*columns], (columns)*sizeof(int), cudaMemcpyDeviceToHost, st);
    }

    return matrix;
}

int *  MatrixFreeOnDevice(int ** matrix, int rows, int columns){
    int *deviceMatrix;
    for(int i=0; i<rows; i++) {
        cudaFree (&deviceMatrix[i*rows]);
    }
    return deviceMatrix;
}


int main()
{
    lwe_instance lwe = GenerateLweInstance(2);

    int * t = GenerateVector(lwe.n, lwe);
    int * secretKey = SecretKeyGen(t, lwe);

    int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    int ** C1 = Encrypt(30, publicKey, lwe);
    int ** C2 = Encrypt(15, publicKey, lwe);

    cudaStream_t st;
    cudaStreamCreate(&st);

    int rows = 5;
    int columns = 5;

    int ** sample1  = GenerateIdentity(rows, columns);


    // // printMatrix(sample1, rows, columns, "sample1");
    int * deviceM1 = MatrixAllocOnDevice(sample1, rows, columns, st);

    // int ** sample2  = GenerateIdentity(rows, columns*4);

    // printMatrix(sample2, rows, columns, "sample2");
    int * deviceM2 = MatrixAllocOnDevice(GenerateMatrixOverQ(rows, columns*4, 4), rows, columns*4, st);

    printMatrix(sample1, rows, columns, "deviceM1'");

    // int ** mult = MultiplyMatrixxMatrixOverQ(sample1, sample2, rows, columns, rows, columns, 100000);

    

    // -----------------------------------------

    int numberOfThreads = rows * columns;
    int requiredNumberOfThreadsPerBlock = 1;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if(numberOfThreads > THREADLIMITPERBLOCK) {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = (numberOfThreads / pw) + 1;
    }

    dim3 grid(requiredNumberOfThreadsPerBlock, 1, 1);
    dim3 block(pw, 1, 1);

    printf("[%d][%d]", requiredNumberOfThreadsPerBlock, pw);

    int l = 4;

    // PrintMatrix<<<grid, block, 0, st>>>(deviceM1, columns);
    // MatSum<<<grid, block, 0, st>>>(deviceM1, deviceM2, deviceM3, numberOfThreads, rows); // total, row_length
    // MatMultiplication<<<grid, block, 0, st>>>(deviceM3, deviceM3, deviceM3, numberOfThreads, rows); // result_matrix_length, row_length
    // PrintMatrix<<<grid, block, 0, st>>>(deviceM1, numberOfThreads);
    GPUBitDecomp<<< grid, block, 0, st>>>(deviceM1, deviceM2, rows * columns, l);
    int ** m3 = MatrixAllocOnHost(deviceM2, rows, columns*4, st);


    numberOfThreads = rows * columns;
    requiredNumberOfThreadsPerBlock = 1;
    closestPowerOf2 = (log2(numberOfThreads) + 1);
    pw = 1 << closestPowerOf2;

    if(numberOfThreads > THREADLIMITPERBLOCK) {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = (numberOfThreads / pw) + 1;
    }

    dim3 grid1(requiredNumberOfThreadsPerBlock, 1, 1);
    dim3 block1(pw, 1, 1);


    printf("R: %d  %d %d", requiredNumberOfThreadsPerBlock, pw, rows * columns);
    GPUBitDecompInverse<<< grid1, block1, 0, st>>>(deviceM2, deviceM1, rows * columns, l);
    // PrintMatrix<<<grid, block, 0, st>>>(deviceM3, numberOfThreads*4);

    int ** m2 = MatrixAllocOnHost(deviceM1, rows, columns, st);
    printMatrix(m2, rows, columns, "BotDecompInverse");

    // CHECK_LAST_CUDA_ERROR();

    cudaStreamSynchronize(st);

    printMatrix(m3, rows, columns*4, "BitDecomp");
    printMatrix(m2, rows, columns, "BitDecompInverse");

    return 0;
}