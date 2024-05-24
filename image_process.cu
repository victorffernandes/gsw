#include "lib/acc_gsw.cu"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


const int MAX_CONCURRENT = 500;


void xor_image_cpu(cbyte* img_1_cPixels, cbyte* img_2_cPixels, cbyte * result_img_cPixels, int image_size, lwe_instance lwe) {

        for (int j = 0; j < image_size; j++)
        {
                result_img_cPixels[j] = ByteXOR(img_1_cPixels[j], img_2_cPixels[j], lwe);
        }
}

void xor_image_gpu(cbyte* img_1_cPixels, cbyte* img_2_cPixels, cbyte * result_img_cPixels, int image_size, lwe_instance lwe) {
    int total_streams = image_size * BYTE_LENGTH;
    cudaStream_t st[total_streams+1];

    printf(" %d ", image_size);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for(int step = 0; step < image_size; step+= MAX_CONCURRENT){
        int nextLimit = step+MAX_CONCURRENT;
        if(nextLimit > image_size) nextLimit = image_size;
        for (int i = step; i < nextLimit; i++)
        {
            int*** c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);

            for (int j = 0; j < BYTE_LENGTH; j++)
            {
                int st_index = i + BYTE_LENGTH * j;
                CHECK_CUDA_ERROR(cudaStreamCreate(&st[st_index]));

                int * d_enc = GPUHomomorphicXOR(img_1_cPixels[i][j], img_2_cPixels[i][j], lwe, st[st_index]);

                c[j] = GenerateEmpty(lwe.N, lwe.N);
                MatrixAllocOnHostAsync(d_enc, c[j], lwe.N, lwe.N, st[st_index]);
                CHECK_CUDA_ERROR(cudaStreamDestroy(st[st_index]));
            }
            result_img_cPixels[i] = c;
        }
        cudaDeviceSynchronize();

        for (int i = step; i < nextLimit; i++)
        {
            for (int j = 0; j < BYTE_LENGTH; j++)
            {
                FreeMatrix(img_1_cPixels[i][j], lwe.N);
                FreeMatrix(img_2_cPixels[i][j], lwe.N);
            }
        }
    }
}

int main(int argc, char *argv[])
{
        if (argc < 3)
        {
                printf("Usage: %s <origin_file_name_1> <origin_file_name_2> <target_file_name> \n", argv[0]);
                return 1;
        }

        char *f1 = argv[1];
        char *f2 = argv[2];
        char *r = argv[3];

        clock_t start = clock();

        BMPHeader header1;
        cbyte * img_1_cPixels = read_cbmp(f1, &header1);

        BMPHeader header2;
        cbyte * img_2_cPixels = read_cbmp(f2, &header2);


        if (header1.width != header2.width || 
                header1.height != header2.height || 
                header1.bits_per_pixel != header2.bits_per_pixel || 
                header1.image_size != header2.image_size || 
                header1.lambda != header2.lambda)
        {
                free(img_1_cPixels);
                free(img_2_cPixels);
                printf("Images are not compatible for processing\n");
                return 1;
        }
        

        cbyte * result_img_cPixels = (cbyte *)malloc(sizeof(cbyte) * header1.image_size);
        lwe_instance lwe = GenerateLweInstance(header1.lambda);

        if (is_gpu_enabled()) 
        {
                xor_image_gpu(img_1_cPixels, img_2_cPixels, result_img_cPixels, header1.image_size, lwe);
        }
        else 
        {
                xor_image_cpu(img_1_cPixels, img_2_cPixels, result_img_cPixels, header1.image_size, lwe);
        }

        write_cbmp(r, &header1, result_img_cPixels, lwe);
       

        clock_t end = clock();

        printf("\n time: %f ",(double)(end - start)/ CLOCKS_PER_SEC);

        free(img_1_cPixels);
        free(img_2_cPixels);
        free(result_img_cPixels);
        return 0;
}