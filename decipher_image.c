#include "lib/gsw.c"
#include "utils/bmp.c"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/helpers.c"


int main(int argc, char *argv[])
{
        if (argc < 5)
        {
                printf("Usage: %s <origin_file_name>  <target_file_name> <secret-key-vector>\n", argv[0]);
                return 1;
        }

        char *f = argv[2];
        char *f_ = argv[3];

        BMPHeader header;
        cbyte * img_1_cPixels = read_cbmp(f, &header);
        lwe_instance lwe = GenerateLweInstance(header.lambda);

        int *t = GenerateVector(lwe.n, lwe);

        if(!formatKey(argv[4], t, lwe.n)){
                printf("Invalid key format\n");
                printf("Usage: %s <lambda> <origin_file_name> <target_file_name> <secret-key-vector> \n");
                return 1;
        }

        int *secretKey = SecretKeyGen(t, lwe);
        int *v = Powersof2(secretKey, lwe);

        int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]


        uint8_t *data_ = (uint8_t *)malloc(sizeof(uint8_t) * header.image_size);

        for (int j = 0; j < header.image_size; j++)
        {
                data_[j] = ByteDecrypt(img_1_cPixels[j], v, lwe);
        }

        write_bmp(f_, header, data_);
        free(data_);
        free(img_1_cPixels);
        free(t);
        free(secretKey);
        free(v);
        free(publicKey);
}