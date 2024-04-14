#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
        if (argc < 3)
        {
                printf("Usage: %s <origin_file_name_1> <origin_file_name_2> <target_file_name> \n", argv[0]);
                return 1;
        }

        char *f1 = argv[1];
        char *f2 = argv[2];
        char *r = argv[3];

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

        for (int j = 0; j < header1.image_size; j++)
        {
                printf(" [%d]", j);
                result_img_cPixels[j] = ByteXOR(img_1_cPixels[j], img_2_cPixels[j], lwe);
        }

        write_cbmp(r, &header1, result_img_cPixels, lwe);
        // printf("------------------------------------ \n");

        free(img_1_cPixels);
        free(img_2_cPixels);
        free(result_img_cPixels);
        return 0;
}