#include <stdio.h>
#include <stdlib.h>
#include "gsw.c"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define THREADLIMITPERBLOCK 512 

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

dim3 getBlockForMatrix(int rows, int columns){
    int numberOfThreads = rows * columns;
    int requiredNumberOfThreadsPerBlock = 512;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if(numberOfThreads > THREADLIMITPERBLOCK) {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = ceil ((float)numberOfThreads / (float)pw)  ;
    }

    dim3 block(pw, 1, 1);
    printf("block [%d] ", pw);

    return block;
}

dim3 getGridForMatrix(int rows, int columns){
    int numberOfThreads = rows * columns;
    int requiredNumberOfThreadsPerBlock = 1;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if(numberOfThreads > THREADLIMITPERBLOCK) {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = ceil ( (float)numberOfThreads / (float)pw)  ;
    }

    dim3 grid(requiredNumberOfThreadsPerBlock, 1, 1);
    printf("grid [%d] ", requiredNumberOfThreadsPerBlock);

    return grid;
}

__global__ void PrintMatrix(int * m1, int total) {
    int id = threadIdx.x + blockIdx.x * blockDim.x; 
    if(id < total){
        for (int i = 0; i < total; i++){
        printf(" %d", m1[i]);
        }
    }
}

__global__ void MatMultiplication(int * lm1, int * cm2, int * rm3, int row, int column, int width, lwe_instance lwe) { // [4,3] [3,5]
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ci = (id % column);
    int ri = (id / column);
    int total = row * column;
    if(id < total){
        printf("id: %d  \n", id);
        int sum = 0;
        for(int i = 0; i < width; ++i){
            sum += lm1[ri*width + i] * cm2[column*i + ci];
        }
        
        rm3[id] = sum % lwe.q;
    }
}

__global__ void MatSum(int * lm1, int * cm2, int * rm3, int total) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < total){
        rm3[id] = (lm1[id] + cm2[id]);
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
            rv[id] += (1 << i) * bv[id*l + i];
        }
    }
}

__global__ void GPUBitDecomp(int * v, int * rv, int total, int l) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < total){
        //printf(" [%d %d %d]", v[id], id*l, id*l + 3);
        for(int i = 0; i < l; i++){
            rv[id*l + i] = (v[id] >> i) & 1;
        }
    }
}

__global__ void GPUGenerateR(int * device_r, int size, int seed ){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandState s;

    // seed a random number generator
    curand_init(seed, size, id, &s);
    int a = curand(&s);
    for (int i = 0;  i < 32 && i < size; i++) 
    {
        device_r[id*32+i] = (a >> i) & 1;
    }
}


void GPUFlatten(int * device_lin_mat, int * device_lin_mat_temp, lwe_instance lwe, cudaStream_t st){
    dim3 grid = getGridForMatrix(lwe.N, lwe.n + 1);
    dim3 block = getBlockForMatrix(lwe.N, lwe.n + 1);

    GPUBitDecompInverse<<< grid, block, 0, st>>>(device_lin_mat, device_lin_mat_temp, lwe.N * lwe.N, lwe.l); // [N, N] -> [N, n+1]

    GPUBitDecomp<<< grid, block, 0, st>>>(device_lin_mat_temp, device_lin_mat, lwe.N * (lwe.n + 1), lwe.l); // [N, n + 1] -> [N, N]
}

int *  MatrixAllocOnDevice(int ** matrix, int rows, int columns, cudaStream_t st){
    int * deviceMatrix;
    CHECK_CUDA_ERROR(cudaMallocAsync(&deviceMatrix,  rows * columns *sizeof(int), st));
    for(int i=0; i<rows; i++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync (&deviceMatrix[i*columns], matrix[i], (columns)*sizeof(int), cudaMemcpyHostToDevice, st));
    }

    return deviceMatrix;
}

int *  MatrixAllocEmptyOnDevice(int rows, int columns, cudaStream_t st){
    int * deviceMatrix = nullptr;
    
    CHECK_CUDA_ERROR(cudaMallocAsync(&deviceMatrix,  rows * columns *sizeof(int), st));
    return deviceMatrix;
}

int **  MatrixAllocOnHost(int * deviceMatrix, int rows, int columns, cudaStream_t st){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for(int i=0; i<rows; i++) {
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        CHECK_CUDA_ERROR(cudaMemcpyAsync (matrix[i], &deviceMatrix[i*columns], (columns)*sizeof(int), cudaMemcpyDeviceToHost, st));
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

void ValidateRADecomp(int ** pubKey, lwe_instance lwe, cudaStream_t st, int * v){
    // BitDecomp (R * A)
    int ** R = GenerateBinaryMatrix(lwe.N,lwe.m); // [N, m]

    int * device_pubKey = MatrixAllocOnDevice(pubKey, lwe.m, lwe.n + 1, st); // [m, n+1]
    int * device_R = MatrixAllocOnDevice(R, lwe.N, lwe.m, st); // [m, n+1]


    dim3 grid = getGridForMatrix(lwe.N, lwe.n + 1);
    dim3 block = getBlockForMatrix(lwe.N, lwe.n + 1);
    int * device_RA = MatrixAllocEmptyOnDevice(lwe.N, lwe.n + 1, st); // [N, n+1]
    MatMultiplication<<<grid, block, 0, st>>>(device_R, device_pubKey, device_RA, lwe.N, (lwe.n + 1), lwe.m, lwe);

    cudaStreamSynchronize(st);

    int ** host_RA = MatrixAllocOnHost(device_RA, lwe.N, lwe.n+1, st);
    int ** RA = MultiplyMatrixxMatrixOverQ(R, pubKey, lwe.N, lwe.m, lwe.m, lwe.n+1, lwe.q); // [N, n+1]
    printMatrix(host_RA, lwe.N, lwe.n+1, "GPURA");
    printMatrix(RA, lwe.N, lwe.n+1, "SequentialRA");
    
    int * checkSUM = MultiplyVectorxMatrixOverQ(v, host_RA, lwe.N, lwe.n+1, lwe.q); // [N]
    printVector(checkSUM, lwe.N, "check");
}

int ** GPUEnc(int message, int ** pubKey, lwe_instance lwe, cudaStream_t st){
    int * device_R = MatrixAllocEmptyOnDevice(lwe.N, lwe.m, st); // [N, m]

    dim3 grid = getGridForMatrix(lwe.N, lwe.m);
    dim3 block = getBlockForMatrix(lwe.N, lwe.m);
    GPUGenerateR<<<grid, block, 0 , st>>>(device_R, lwe.N * lwe.m, time(NULL));

    int * device_pubKey = MatrixAllocOnDevice(pubKey, lwe.m, lwe.n + 1, st); // [m, n+1]

    cudaStreamSynchronize(st);

    grid = getGridForMatrix(lwe.N, lwe.n + 1);
    block = getBlockForMatrix(lwe.N, lwe.n + 1);
    int * device_RA = MatrixAllocEmptyOnDevice(lwe.N, lwe.n + 1, st); // [N, n+1]
    MatMultiplication<<<grid, block, 0, st>>>(device_R, device_pubKey, device_RA, lwe.N, (lwe.n + 1), lwe.m, lwe);


    PrintMatrix<<<1,1,0,st>>>(device_RA, lwe.N * (lwe.n + 1));

    grid = getGridForMatrix(lwe.N, lwe.n + 1);
    block = getBlockForMatrix(lwe.N, lwe.n + 1);

    int * device_RABitDecomp = MatrixAllocEmptyOnDevice(lwe.N, lwe.N, st); // [N, m]
    GPUBitDecomp<<< grid, block, 0, st>>>(device_RA, device_RABitDecomp, lwe.N * (lwe.n + 1), lwe.l); // [N, n + 1] -> [N, N]

    int ** m2 = MatrixAllocOnHost(device_RA, lwe.N, lwe.n+1, st);

    printMatrix(m2, lwe.N, lwe.n + 1, "m2");

    // // m * In
    int ** Identity = GenerateIdentity(lwe.N, lwe.N);
    int ** mIdentity = MultiplyMatrixEscalarOverQ(message, Identity, lwe.N, lwe.N, lwe.q); // [N, N]
    int * device_mIdentity = MatrixAllocOnDevice(mIdentity, lwe.N, lwe.N, st); // [m, n+1]

    int * device_Cipher = MatrixAllocEmptyOnDevice(lwe.N, lwe.N, st); // [N, m]

    grid = getGridForMatrix(lwe.N, lwe.N );
    block = getBlockForMatrix(lwe.N, lwe.N );

    MatSum<<<grid, block, 0, st>>>(device_mIdentity, device_RABitDecomp, device_Cipher, lwe.N * lwe.N);

    GPUFlatten(device_Cipher, device_mIdentity, lwe, st);

    cudaStreamSynchronize(st);

    int ** m3 = MatrixAllocOnHost(device_Cipher, lwe.N, lwe.N, st);

    printMatrix(m3, lwe.N, lwe.N, "Cipher");

    cudaStreamSynchronize(st);

    return m3;

    // int ** m3 = MatrixAllocOnHost(device_RA, lwe.N, lwe.n+1, st);

    // printMatrix(m2, lwe.m, lwe.n + 1, "pubKeyla");
    // printMatrix(m1, lwe.N, lwe.m, "R");
    // printMatrix(m3, lwe.N, lwe.n + 1, "RA");
}


int main()
{
    lwe_instance lwe = GenerateLweInstance(10);

    int * t = GenerateVector(lwe.n, lwe);
    int * secretKey = SecretKeyGen(t, lwe);
    int * v = Powersof2(secretKey, lwe);

    int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    int ** C1 = Encrypt(30, publicKey, lwe);
    int ** C2 = Encrypt(15, publicKey, lwe);

    cudaStream_t st;
    cudaStreamCreate(&st);

    int rows = 5;
    int columns = 5;

    int ** sample1  = GenerateIdentity(lwe.N, lwe.m);
    int ** sample2  = GenerateIdentity(lwe.m, lwe.N);

    // printMatrix(sample1, lwe.N, lwe.m, "sample1");
    // printMatrix(sample2,lwe.m, lwe.N, "sample2");
    int * deviceM1 = MatrixAllocOnDevice(sample1, lwe.N, lwe.m, st);
    int * deviceM2 = MatrixAllocOnDevice(sample2, lwe.m, lwe.N, st);
    int * deviceM3 = MatrixAllocEmptyOnDevice(lwe.N, lwe.N, st);

    // GPUFlatten(deviceM1, deviceM2, lwe, st);
    // dim3 grid = getGridForMatrix(lwe.N, lwe.N);
    // dim3 block = getBlockForMatrix(lwe.N, lwe.N);
    // GPUGenerateR<<<grid, block, 0, st>>>(deviceM1, lwe.N * lwe.N, time(NULL));
    int ** cipher = GPUEnc(10, publicKey, lwe, st);

    // printMatrix(cipher, lwe.N, lwe.n+1, "cipher");

    cudaStreamSynchronize(st);

    // int a = Decrypt(cipher, v, lwe);

    int d = MPDecrypt(cipher, v, lwe);
    // printf("Decipher: %d ", a);
    // PrintMatrix<<<grid, block, 0, st>>>(deviceM1, columns);
    // MatSum<<<grid, block, 0, st>>>(deviceM1, deviceM2, deviceM3, numberOfThreads, rows); // total, row_length
    // MatMultiplication<<<grid, block, 0, st>>>(deviceM1, deviceM2, deviceM3, lwe.N, lwe.N, lwe.m); // result_matrix_length, row_length
    // PrintMatrix<<<grid, block, 0, st>>>(deviceM1, numberOfThreads);
    cudaStreamSynchronize(st);
    int ** m3 = MatrixAllocOnHost(deviceM3, lwe.N, lwe.N, st);
    
    // printMatrix(m3, lwe.N, lwe.N, "m3");
    CHECK_LAST_CUDA_ERROR();

    return 0;
}