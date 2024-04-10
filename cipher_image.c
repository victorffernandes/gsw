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
        if (argc < 4)
        {
                printf("Usage: %s <lambda> <origin_file_name> <target_file_name>\n", argv[0]);
                return 1;
        }

        int lambda = atoi(argv[1]);
        char *f = argv[2];
        char *f_ = argv[3];
        char *should_use_key = argv[4];

        lwe_instance lwe = GenerateLweInstance(lambda);
        int *t = GenerateVector(lwe.n, lwe);

        if (should_use_key == "--use-key")
        {
                if(!formatKey(argv[5], t, lwe.n)){
                        printf("Invalid key format\n");
                        printf("Usage: %s <lambda> <origin_file_name> <target_file_name> --use-key <secret-key-vector> \n");
                        return 1;
                }
        }

        char *t_s = argv[4];

        int *secretKey = SecretKeyGen(t, lwe);
        int *v = Powersof2(secretKey, lwe);

        int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

        BMPHeader header;

        uint8_t *data = read_bmp(f, &header);

        cbyte *img_1_cPixels = (cbyte *)malloc(sizeof(cbyte) * header.image_size);

        for (int j = 0; j < header.image_size; j++)
        {
                img_1_cPixels[j] = ByteEncrypt(data[j], publicKey, lwe);
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