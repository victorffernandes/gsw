#include "lib/acc_gsw.cu"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/helpers.c"

const int MAX_CONCURRENT = 8000;

void encrypt_image(cbyte* img_1_cPixels, uint8_t* data, int image_size, int** publicKey, lwe_instance lwe)
{

    for (int j = 0; j < image_size; j++)
    {
        img_1_cPixels[j] = ByteEncrypt(data[j], publicKey, lwe);
    }
}

#endif

#ifdef __CUDACC__
void encrypt_image(cbyte* img_1_cPixels, uint8_t* data, int image_size, int** publicKey, lwe_instance lwe)
{
    int* d_pub = MatrixAllocOnDevice(publicKey, lwe.m, lwe.n + 1);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int total_streams = image_size * BYTE_LENGTH;
    cudaStream_t* st = (cudaStream_t*)malloc(sizeof(cudaStream_t) * total_streams);

    for (int step = 0; step < image_size; step += MAX_CONCURRENT)
    {
        int nextLimit = step + MAX_CONCURRENT;
        if (nextLimit > image_size)
            nextLimit = image_size;
        for (int i = step; i < nextLimit; i++)
        {
            int*** c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);

            for (int j = 0; j < BYTE_LENGTH; j++)
            {
                int st_index = i + BYTE_LENGTH * j;
                CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&st[st_index], cudaStreamNonBlocking));

                int p = (data[i] >> j) & 1;
                int* d_enc = GPUEnc(p, d_pub, lwe, st[st_index]);

                c[j] = GenerateEmpty(lwe.N, lwe.N);
                MatrixAllocOnHostAsync(d_enc, c[j], lwe.N, lwe.N, st[st_index]);
                // PrintStream<<<1,1,0,st[st_index]>>>(i * j);

                CHECK_CUDA_ERROR(cudaStreamDestroy(st[st_index]));
            }
            img_1_cPixels[i] = c;
        }
        cudaDeviceSynchronize();
    }
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <lambda> <origin_file_name> <target_file_name>\n", argv[0]);
        return 1;
    }

    int lambda = atoi(argv[1]);
    char* f = argv[2];
    char* f_ = argv[3];
    char* should_use_key = argv[4];
    char* runOn = argv[6];

    lwe_instance lwe = GenerateLweInstance(SECURITY_64);
    int* t = GenerateVector(lwe.n, lwe);

    if (should_use_key != NULL && strcmp(should_use_key, "--use-key") == 0)
    {
        if (!formatKey(argv[5], t, lwe.n))
        {
            printf("Invalid key format\n");
            printf("Usage: <lambda> <origin_file_name> <target_file_name> --use-key <secret-key-vector> \n");
            return 1;
        }
    }

    clock_t start = clock();

    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    BMPHeader header;

    uint8_t* data = read_bmp(f, &header);

    cbyte* img_1_cPixels = (cbyte*)malloc(sizeof(cbyte) * header.image_size);

    if (strcmp(runOn, "gpu") == 0 && is_gpu_enabled())
    {
        encrypt_image_gpu(img_1_cPixels, data, header.image_size, publicKey, lwe);
    }
    else
    {
        encrypt_image_cpu(img_1_cPixels, data, header.image_size, publicKey, lwe);
    }

    write_cbmp(f_, &header, img_1_cPixels, lwe);
    printVector(t, lwe.n, (char*)"t: ");

    clock_t end = clock();

    printf(" time: %f ", (double)(end - start) / CLOCKS_PER_SEC);

    free(img_1_cPixels);
    free(data);
    free(t);
    free(v);
    free(secretKey);
    free(publicKey);
}