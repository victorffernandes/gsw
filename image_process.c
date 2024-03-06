#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main()
{
        lwe_instance lwe = GenerateLweInstance(30);

        int * t = GenerateVector(lwe.n, lwe);
        int * secretKey = SecretKeyGen(t, lwe);
        int * v = Powersof2(secretKey, lwe);
        
        int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

        byte *pixels_img_1;
        int width_img_1;
        int height_img_1;
        int bytesPerPixel_img_1;
        byte *pixels_img_2;
        int width_img_2;
        int height_img_2;
        int bytesPerPixel_img_2;
        byte *pixels_img_3 = (byte*)malloc(width_img_2*bytesPerPixel_img_2*height_img_2);


        ReadImage("resources/img1_10x10.bmp", &pixels_img_1, &width_img_1, &height_img_1,&bytesPerPixel_img_1);
        ReadImage("resources/img2_10x10.bmp", &pixels_img_2, &width_img_2, &height_img_2,&bytesPerPixel_img_2);
        printf("img1 size: %d %d total: %d ", width_img_1, height_img_1, width_img_1 * height_img_1);
        printf("img2 size: %d %d total: %d ", width_img_2, height_img_2, width_img_2 * height_img_2);

        cbyte * img_1_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width_img_1 * height_img_1 * bytesPerPixel_img_1);
        cbyte * img_2_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width_img_2 * height_img_2 * bytesPerPixel_img_2);
        cbyte * img_3_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width_img_2 * height_img_2 * bytesPerPixel_img_2);

        for(int i = 0; i < width_img_1*bytesPerPixel_img_1; i++){
                for(int j = 0; j < height_img_1; j++){
                        img_1_cPixels[i * width_img_1 + j] = ByteEncrypt(pixels_img_1[i * width_img_1 + j], publicKey, lwe);
                }
                printf("finished %d row \n", i);
        }
        printf("first image");
        for(int i = 0; i < width_img_2*bytesPerPixel_img_2; i++){
                for(int j = 0; j < height_img_2; j++){
                        img_2_cPixels[i * width_img_2 + j] = ByteEncrypt(pixels_img_2[i * width_img_2 + j], publicKey, lwe);
                }
                printf("finished %d row \n", i);
        }

        printf("second image");

        for(int i = 0; i < width_img_2*bytesPerPixel_img_2; i++){
                for(int j = 0; j < height_img_2; j++){
                   img_3_cPixels[i * width_img_2 + j] = ByteXOR(img_1_cPixels[i * width_img_1 + j], img_2_cPixels[i * width_img_2 + j],v,  lwe);
                   pixels_img_3[i * width_img_2 + j] = ByteDecrypt(img_3_cPixels[i * width_img_2 + j], v, lwe);
                }
                printf("finished %d row \n", i);
        }
        WriteImage("img3_10x10.bmp", pixels_img_3, width_img_2, height_img_2, bytesPerPixel_img_2);
        // free(pixels_img_1);
        // free(pixels_img_2);
        return 0;
}