#include <stdio.h>
#include <stdlib.h>
#include "gsw.c"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <time.h>

#define THREADLIMITPERBLOCK 512

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "__________CUDA_ERROR_____________" << std::endl;
        fprintf(stderr, "CUDA ERROR: %s %s %d\n",
            cudaGetErrorString(err), file, line);
        if (abort)
        {
            exit(err);
        }
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "__________CUDA_ERROR_____________" << std::endl;
        fprintf(stderr, "CUDA ERROR: %s %s %d\n",
            cudaGetErrorString(err), file, line);
        if (abort)
        {
            exit(err);
        }
    }
}

dim3 getBlockForMatrix(int rows, int columns)
{
    int numberOfThreads = rows * columns;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if (numberOfThreads > THREADLIMITPERBLOCK)
    {
        pw = THREADLIMITPERBLOCK;
    }

    dim3 block(pw, 1, 1);

    return block;
}

dim3 getGridForMatrix(int rows, int columns)
{
    int numberOfThreads = rows * columns;
    int requiredNumberOfThreadsPerBlock = 1;
    int closestPowerOf2 = (log2(numberOfThreads) + 1);
    int pw = 1 << closestPowerOf2;

    if (numberOfThreads > THREADLIMITPERBLOCK)
    {
        pw = THREADLIMITPERBLOCK;
        requiredNumberOfThreadsPerBlock = ceil((float)numberOfThreads / (float)pw);
    }

    dim3 grid(requiredNumberOfThreadsPerBlock, 1, 1);

    return grid;
}

__global__ void PrintMatrix(int* matrix, int total)
{
    for (int i = 0; i < total; i++)
    {
        printf(" %d ", matrix[i]);
    }
}

__global__ void PrintStream(int id)
{
    printf(" %d ", id);
}

__global__ void MatMultiplication(int* lm1, int* cm2, int* rm3, int row, int column, int width, lwe_instance lwe)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ci = (id % column);
    int ri = (id / column);
    int total = row * column;

    if (id < total)
    {
        int sum = 0;
        for (int i = 0; i < width; ++i)
        {
            sum += lm1[ri * width + i] * cm2[column * i + ci];
        }

        rm3[id] = sum % lwe.q;
    }
}

__global__ void MatSum(int* lm1, int* cm2, int total)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < total)
    {
        lm1[id] = (lm1[id] + cm2[id]);
    }
}

__global__ void MatSub(int* lm1, int* cm2, int* rm3, int total, int width)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int ri = id % width;
    int ci = (id / width);
    if (id < total)
    {
        rm3[ri * width + ci] = (lm1[ri * width + ci] - cm2[ri * width + ci]);
    }
}

__global__ void GPUBitDecompInverse(int* bv, int* rv, int total, int l)
{ // total must not be the number of bits, but the number of elements in rv
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < total)
    {
        rv[id] = 0;
        for (int i = 0; i < l; i++)
        {
            rv[id] += (1 << i) * bv[id * l + i];
        }
    }
}

__global__ void GPUBitDecomp(int* v, int* rv, int total, int l)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < total)
    {
        for (int i = 0; i < l; i++)
        {
            rv[id * l + i] = (v[id] >> i) & 1;
        }
    }
}

__global__ void GPUGenerateR(int* device_r, int size, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int s[];

    if (id % 32 == 0)
    {
        curandState state;

        curand_init(seed, size, id, &state);
        int a = curand(&state);
        s[id / 32] = a;
    }

    __syncthreads();
    device_r[id] = (s[id / 32] >> (id % 32)) & 1;
}


__global__ void GPUGenerateMessageIdentity(int* device_message_identity, int row, int column, int message)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x; //

    if (id < row * column)
    {
        if (id % (column + 1) == 0)
        {
            device_message_identity[id] = message;
        }
        else
        {
            device_message_identity[id] = 0;
        }
    }
}

void GPUFlatten(int* device_lin_mat, int* device_lin_mat_temp, lwe_instance lwe, cudaStream_t st)
{
    dim3 grid = getGridForMatrix(lwe.N, lwe.n + 1);
    dim3 block = getBlockForMatrix(lwe.N, lwe.n + 1);

    GPUBitDecompInverse <<<grid, block, 0, st>>> (device_lin_mat, device_lin_mat_temp, lwe.N * (lwe.n+1), lwe.l); // [N, N] -> [N, n+1]

    GPUBitDecomp <<<grid, block, 0, st>>> (device_lin_mat_temp, device_lin_mat, lwe.N * (lwe.n + 1), lwe.l); // [N, n + 1] -> [N, N]
}

int* MatrixAllocOnDeviceAsync(int** matrix, int rows, int columns, cudaStream_t st)
{
    int* deviceMatrix;
    int* m_vector = (int*)malloc(sizeof(int) * rows * columns);
    CHECK_CUDA_ERROR(cudaMallocAsync(&deviceMatrix, rows * columns * sizeof(int), st));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++){
            m_vector[i * rows + j] = matrix[i][j];
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(deviceMatrix, m_vector, (columns * rows) * sizeof(int), cudaMemcpyHostToDevice, st));

    return deviceMatrix;
}

void MatrixAllocOnHostAsync(int* deviceMatrix, int ** hostMatrix, int rows, int columns, cudaStream_t st) {

    int * m_vector = (int*)malloc(sizeof(int) * rows * columns);
    

    cudaMemcpyAsync(m_vector, deviceMatrix, (columns * rows) * sizeof(int), cudaMemcpyDeviceToHost, st);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++){
            hostMatrix[i][j] = m_vector[i * rows + j];
        }
    }

    // CHECK_CUDA_ERROR(cudaFreeAsync(deviceMatrix, st));
}

int* MatrixAllocEmptyOnDeviceAsync(int rows, int columns, cudaStream_t st)
{
    int* deviceMatrix;
    CHECK_CUDA_ERROR(cudaMallocAsync(&deviceMatrix, rows * columns * sizeof(int), st));

    return deviceMatrix;
}

int* MatrixAllocOnDevice(int** matrix, int rows, int columns)
{
    int* deviceMatrix;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceMatrix, rows * columns * sizeof(int)));
    for (int i = 0; i < rows; i++)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(&deviceMatrix[i * columns], matrix[i], (columns) * sizeof(int), cudaMemcpyHostToDevice));
    }

    return deviceMatrix;
}

int* MatrixAllocEmptyOnDevice(int rows, int columns)
{
    int* deviceMatrix;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceMatrix, rows * columns * sizeof(int)));

    return deviceMatrix;
}

int* MatrixFreeOnDevice(int** matrix, int rows, int columns)
{
    int* deviceMatrix;
    for (int i = 0; i < rows; i++)
    {
        cudaFree(&deviceMatrix[i * rows]);
    }
    return deviceMatrix;
}

void getMemInfo()
{
    size_t freeMem, totalMem;
    cudaError_t cudaStatus = cudaMemGetInfo(&freeMem, &totalMem);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemGetInfo falhou: %s\n", cudaGetErrorString(cudaStatus));
    }

    printf("Total Memory: %zu bytes\n", totalMem);
    printf("Free Memory: %zu bytes\n", freeMem);
}

int* GPUHomomorphicXOR(int ** C1, int ** C2, lwe_instance lwe, cudaStream_t st)
{
    dim3 gridNN = getGridForMatrix(lwe.N, lwe.N);
    dim3 blockNN = getBlockForMatrix(lwe.N, lwe.N);

    int* device_C1 = MatrixAllocOnDeviceAsync(C1, lwe.N, lwe.N, st); // [N, m]
    int* device_C2 = MatrixAllocOnDeviceAsync(C2, lwe.N, lwe.N, st); // [N, m]

    MatSum <<<gridNN, blockNN, 0, st>>> (device_C1, device_C2, lwe.N * lwe.N);
    GPUFlatten(device_C1, device_C2, lwe, st);

    CHECK_CUDA_ERROR(cudaFreeAsync(device_C2, st));
    return device_C1;
}

int* GPUEnc(int message, int* device_pubKey, lwe_instance lwe, cudaStream_t st)
{
    dim3 gridNm = getGridForMatrix(lwe.N, lwe.m);
    dim3 blockNm = getBlockForMatrix(lwe.N, lwe.m);

    dim3 gridNn1 = getGridForMatrix(lwe.N, lwe.n + 1);
    dim3 blockNn1 = getBlockForMatrix(lwe.N, lwe.n + 1);

    dim3 gridNN = getGridForMatrix(lwe.N, lwe.N);
    dim3 blockNN = getBlockForMatrix(lwe.N, lwe.N);

    // mem-alloc
    int** R = GenerateBinaryMatrix(lwe.N, lwe.m);
    int* device_R = MatrixAllocOnDeviceAsync(R, lwe.N, lwe.m, st); // [N, m]

    // GPUGenerateR<<< gridNm, blockNm, 1024, st>>>(device_R, lwe.N, lwe.m);

    int* device_RA = MatrixAllocEmptyOnDeviceAsync(lwe.N, lwe.n + 1, st);      // [N, n+1]
    int* device_RABitDecomp = MatrixAllocEmptyOnDeviceAsync(lwe.N, lwe.N, st); // [N, N]
    int* device_mIdentity = MatrixAllocEmptyOnDeviceAsync(lwe.N, lwe.N, st);   // [m, n+1]

    GPUGenerateMessageIdentity <<<gridNN, blockNN, 0, st>>> (device_mIdentity, lwe.N, lwe.N, message);

    MatMultiplication <<<gridNn1, blockNn1, 0, st>>> (device_R, device_pubKey, device_RA, lwe.N, (lwe.n + 1), lwe.m, lwe);
    GPUBitDecomp <<<gridNn1, blockNn1, 0, st>>> (device_RA, device_RABitDecomp, lwe.N * (lwe.n + 1), lwe.l); // [N, n + 1] -> [N, N]

    MatSum <<<gridNN, blockNN, 0, st>>> (device_mIdentity, device_RABitDecomp, lwe.N * lwe.N);
    GPUFlatten(device_mIdentity, device_RABitDecomp, lwe, st);

    CHECK_CUDA_ERROR(cudaFreeAsync(device_RABitDecomp, st));
    CHECK_CUDA_ERROR(cudaFreeAsync(device_R, st));
    CHECK_CUDA_ERROR(cudaFreeAsync(device_RA, st));

    return device_mIdentity;
}

int is_gpu_enabled() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount > 0) {
        cudaSetDevice(0);

        printf("GPU Card available, executing using CUDA devices: %d ", deviceCount);
    }
    else {
        printf("GPU Card unavailable, executing using C");
    }

    return deviceCount > 0;
}
