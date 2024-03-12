#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
        lwe_instance lwe = GenerateLweInstance(30);

        int *t = GenerateVector(lwe.n, lwe);
        int *secretKey = SecretKeyGen(t, lwe);
        int *v = Powersof2(secretKey, lwe);

        int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

        byte *pixels_img_1;
        int width;
        int height;
        int bytesPerPixel;
        byte *pixels_img_2;
        byte *pixels_img_3 = (byte *)malloc(width * bytesPerPixel * height);

        ReadImage("resources/10x10-branco.bmp", &pixels_img_1, &width, &height, &bytesPerPixel);
        ReadImage("resources/10x10-00FFDE.bmp", &pixels_img_2, &width, &height, &bytesPerPixel);
        printf("img1 size: %d %d total: %d ", width, height, width * height);
        printf("img2 size: %d %d total: %d ", width, height, width * height);

        cbyte *img_1_cPixels = (cbyte *)malloc(sizeof(cbyte) * width * height * bytesPerPixel);
        cbyte *img_2_cPixels = (cbyte *)malloc(sizeof(cbyte) * width * height * bytesPerPixel);
        cbyte *img_3_cPixels = (cbyte *)malloc(sizeof(cbyte) * width * height * bytesPerPixel);

        int w = ((width) * (bytesPerPixel));
        for (int j = 0; j < height; j++)
        {
                for (int i = 0; i < w; i++)
                {
                        img_1_cPixels[j * w + i] = ByteEncrypt(pixels_img_1[j * w + i], publicKey, lwe);
                        img_2_cPixels[j * w + i] = ByteEncrypt(pixels_img_2[j * w + i], publicKey, lwe);
                        img_3_cPixels[j * w + i] = ByteXOR(img_1_cPixels[j * w + i], img_2_cPixels[j * w + i],v,  lwe);
                        pixels_img_3[j * w + i] = ByteDecrypt(img_3_cPixels[j * w + i], v, lwe);
                        // pixels_img_3[j * w + i] = pixels_img_2[j * w + i];
                }
                printf("finished %d row \n", j);
        }

         WriteImage("img3_10x10.bmp", pixels_img_3, width, height, bytesPerPixel);
        // //WriteImage("img4_10x10.bmp", pixels_img_2, width, height, bytesPerPixel);
        // free(pixels_img_1);
        // free(pixels_img_2);
        // free(pixels_img_3);
        // free(img_1_cPixels);
        // free(img_2_cPixels);
        // free(img_3_cPixels);
        return 0;
}