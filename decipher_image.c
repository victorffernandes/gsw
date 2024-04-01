#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

        char file_name_[50] = "resources/";
        strcat(file_name_, f);
        strcat(file_name_, ".cbmp");


        BMPHeader header;
        cbyte * img_1_cPixels = read_cbmp(file_name_, &header);

        uint8_t *data_ = (uint8_t *)malloc(sizeof(uint8_t) * header.image_size);

        lwe_instance lwe = GenerateLweInstance(header.lambda);
        int *t = GenerateVector(lwe.n, lwe);
        int *secretKey = SecretKeyGen(t, lwe);
        int *v = Powersof2(secretKey, lwe);

        int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

        printf("------------------------------------ \n");
        for (int j = 0; j < header.image_size; j++)
        {
                data_[j] = ByteDecrypt(img_1_cPixels[j], v, lwe);
                printf(" [%d %d]", j, data_[j]);
        }

        char file_name[50] = "resources/";
        strcat(file_name, f);
        strcat(file_name, "_new.bmp");

        write_bmp(file_name, header, data_);
}