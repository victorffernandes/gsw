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
        int width;
        int height;
        int bytesPerPixel;
        byte *pixels_img_2;
        byte *pixels_img_3 = (byte*)malloc(width*bytesPerPixel*height);


        ReadImage("resources/rectangle.bmp", &pixels_img_1, &width, &height,&bytesPerPixel);
        ReadImage("resources/circle.bmp", &pixels_img_2, &width, &height,&bytesPerPixel);
        printf("img1 size: %d %d total: %d ", width, height, width * height);
        printf("img2 size: %d %d total: %d ", width, height, width * height);

        cbyte * img_1_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width * height * bytesPerPixel);
        cbyte * img_2_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width * height * bytesPerPixel);
        cbyte * img_3_cPixels =  (cbyte *) malloc(sizeof(cbyte) * width * height * bytesPerPixel);

        for(int i = 0; i < (int)(4 * ceil((float)width/4.0f))*bytesPerPixel; i++){
                for(int j = 0; j < height; j++){
                        img_1_cPixels[i * width + j] = ByteEncrypt(pixels_img_1[i * width + j], publicKey, lwe);
                        img_2_cPixels[i * width + j] = ByteEncrypt(pixels_img_2[i * width + j], publicKey, lwe);
                        img_3_cPixels[i * width + j] = ByteXOR(img_1_cPixels[i * width + j], img_2_cPixels[i * width + j],v,  lwe);
                        pixels_img_3[i * width + j] = ByteDecrypt(img_2_cPixels[i * width + j], v, lwe);
                }
                printf("finished %d row \n", i);
        }

        WriteImage("img3_10x10.bmp", pixels_img_3, width, height, bytesPerPixel);
        free(pixels_img_1);
        free(pixels_img_2);
        free(pixels_img_3);
        free(img_1_cPixels);
        free(img_2_cPixels);
        free(img_3_cPixels);
        return 0;
}