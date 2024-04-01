#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// int main()
// {
//         lwe_instance lwe = GenerateLweInstance(5);

//         int *t = GenerateVector(lwe.n, lwe);
//         int *secretKey = SecretKeyGen(t, lwe);
//         int *v = Powersof2(secretKey, lwe);

//         int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

//         int width;
//         int height;
//         int bytesPerPixel;
//         int paddedRowSize;
//         int dataOffset;
//         byte *pixels_img_1 = (byte *)malloc(width * bytesPerPixel * height);
//         byte *result_img_2 = (byte *)malloc(width * bytesPerPixel * height);

//         FILE *imageFile = fopen("resources/10x10-00FFDE.bmp", "rb");
//         FILE *resultFile = fopen("resources/result.bmp", "wb");

//         ReadImage(imageFile, &pixels_img_1, &width, &height, &paddedRowSize, &bytesPerPixel);
//         printf("img1 size: %d %d total: %d ", width, height, width * height);
//         printf("img2 size: %d %d total: %d ", width, height, width * height);

//         cbyte *img_1_cPixels = (cbyte *)malloc(sizeof(cbyte) * width * height * bytesPerPixel);
//         cbyte *result_cPixels = (cbyte *)malloc(sizeof(cbyte) * width * height * bytesPerPixel);

//         WriteImageHeader(resultFile, width, height, bytesPerPixel);

//         // int w = ((width) * (bytesPerPixel));
//         // for (int j = 0; j < height; j++)
//         // {
//         //         for (int i = 0; i < w; i++)
//         //         {
//         //                 img_1_cPixels[j * w + i] = ByteEncrypt(pixels_img_1[j * w + i], publicKey, lwe);
//         //                 WriteFileCByte(resultFile, img_1_cPixels[j * w + i], lwe);
//         //                 printf(" %d ", pixels_img_1[j * w + i]);
//         //                 // pixels_img_3[j * w + i] = pixels_img_2[j * w + i];
//         //         }
//         //         printf("finished %d row \n", j);
//         // }
//         img_1_cPixels[0] = ByteEncrypt(pixels_img_1[0], publicKey, lwe);
//         WriteFileCByte(resultFile, img_1_cPixels[0], lwe);
//         fclose(resultFile);

//         resultFile = fopen("resources/result.bmp", "rb");

//         ReadCipherImageHeader(resultFile, &dataOffset, &width, &height, &bytesPerPixel);

//         // for (int j = 0; j < height; j++)
//         // {
//         //         for (int i = 0; i < w; i++)
//         //         {
//         //                 result_cPixels[j * w + i] = ReadFileCByte(resultFile, lwe);
//         //                 result_img_2[j * w + i] = ByteDecrypt(result_cPixels[j * w + i], v, lwe);
//         //                 printf(" %d ", result_img_2[j * w + i]);
//         //         }
//         //         printf("finished %d row \n", j);
//         // }

//         // //WriteImage("img4_10x10.bmp", pixels_img_2, width, height, bytesPerPixel);
//         // free(pixels_img_1);
//         // free(pixels_img_2);
//         // free(pixels_img_3);
//         // free(img_1_cPixels);
//         // free(img_2_cPixels);
//         // free(img_3_cPixels);
//         return 0;
// }

int main(int argc, char *argv[])
{

        printf("Argument count: %d\n", argc);
        printf("Argumentv: %d\n", argv[1]);

        if (argc < 3)
        {
                printf("Usage: %s <lambda> <file_name>  \n", argv[0]);
                return 1;
        }

        int lambda = atoi(argv[1]);
        char * f = argv[2];

        srand(4);
        lwe_instance lwe = GenerateLweInstance(lambda);

        int *t = GenerateVector(lwe.n, lwe);
        int *secretKey = SecretKeyGen(t, lwe);
        int *v = Powersof2(secretKey, lwe);

        int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

        BMPHeader header;
        char file_name[50] = "resources/";
        strcat(file_name, f);
        strcat(file_name, ".bmp");

        uint8_t * data = read_bmp(file_name, &header);

        cbyte *img_1_cPixels = (cbyte *)malloc(sizeof(cbyte) * header.image_size);
        printf("------------------------------------ \n");
        for (int j = 0; j < header.image_size; j++)
        {
                printf(" [%d %d]", j, data[j]);
                img_1_cPixels[j] = ByteEncrypt(data[j], publicKey, lwe);
        }

        char file_name_[50] = "resources/";
        strcat(file_name_, f);
        strcat(file_name_, ".cbmp");

        write_cbmp(file_name_, &header, img_1_cPixels, lwe);
        printVector(t, lwe.n, "t: ");
        free(img_1_cPixels);
}