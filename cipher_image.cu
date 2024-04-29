#include "lib/acc_gsw.cu"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/helpers.c"

void encrypt_image_cpu(cbyte* img_1_cPixels, uint8_t* data, int image_size, int** publicKey, lwe_instance lwe) {

    for (int j = 0; j < image_size; j++)
    {
        img_1_cPixels[j] = ByteEncrypt(data[j], publicKey, lwe);
    }
}

void encrypt_image_gpu(cbyte* img_1_cPixels, uint8_t* data, int image_size, int** publicKey, lwe_instance lwe) {
    int total_streams = image_size * BYTE_LENGTH;
    cudaStream_t st[total_streams];

    int* d_pub = MatrixAllocOnDevice(publicKey, lwe.m, lwe.n + 1);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (int i = 0; i < image_size; i++)
    {
        int*** c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);

        for (int j = 0; j < BYTE_LENGTH; j++)
        {
            int st_index = i + BYTE_LENGTH * j;
            CHECK_CUDA_ERROR(cudaStreamCreate(&st[st_index]));
            int p = (data[i] >> j) & 1;
            int * d_enc = GPUEnc(p, d_pub, lwe, st[st_index]);
            PrintMatrix<<<1,1,0,0>>>(d_enc, lwe.N * lwe.N);
            // c[j] = MatrixAllocOnHost(d_enc, lwe.N, lwe.N, st[st_index]);
        }
    }

    cudaDeviceSynchronize();
    // synchonizeStreams(st, total_streams);
}


int is_gpu_enabled() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount > 0) {
        printf("GPU Card available, executing using CUDA");
    }
    else {
        printf("GPU Card unavailable, executing using C");
    }

    return deviceCount > 0;
}


int main(int argc, char* argv[])
{
    cudaDeviceReset();
    if (argc < 4)
    {
        printf("Usage: %s <lambda> <origin_file_name> <target_file_name>\n", argv[0]);
        return 1;
    }

    int lambda = atoi(argv[1]);
    char* f = argv[2];
    char* f_ = argv[3];
    char* should_use_key = argv[4];

    lwe_instance lwe = GenerateLweInstance(lambda);
    int* t = GenerateVector(lwe.n, lwe);

    if (should_use_key == "--use-key")
    {
        if (!formatKey(argv[5], t, lwe.n)) {
            printf("Invalid key format\n");
            printf("Usage: <lambda> <origin_file_name> <target_file_name> --use-key <secret-key-vector> \n");
            return 1;
        }
    }

    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    BMPHeader header;

    uint8_t* data = read_bmp(f, &header);

    cbyte* img_1_cPixels = (cbyte*)malloc(sizeof(cbyte) * header.image_size);


    if (is_gpu_enabled()) {
        encrypt_image_gpu(img_1_cPixels, data, header.image_size, publicKey, lwe);
    }
    else {
        encrypt_image_cpu(img_1_cPixels, data, header.image_size, publicKey, lwe);
    }


    write_cbmp(f_, &header, img_1_cPixels, lwe);
    printVector(t, lwe.n, "t: ");


    free(img_1_cPixels);
    free(data);
    free(t);
    free(v);
    free(secretKey);
    free(publicKey);
}