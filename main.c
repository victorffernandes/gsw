#include "lib/gsw.c"
#include <time.h>
#include <math.h>


int time_test(int n, int ** pubKey, lwe_instance lwe)
{
    clock_t start, end;
    double cpu_time_used, gpu_time_used;

    start = clock();

    for (int i = 0; i < n; i++)
    {
        Encrypt(1, pubKey, lwe); // C[N, N]
    }

    end = clock();
    gpu_time_used = (((double)(end - start)) / (double)CLOCKS_PER_SEC);

    printf("\n CPU Encrypt: %f \n", gpu_time_used);
}

int main()
{
    lwe_instance lwe = GenerateLweInstance(30);

    int * t = GenerateVector(lwe.n, lwe);
    int * secretKey = SecretKeyGen(t, lwe);
    int * v = Powersof2(secretKey, lwe);
    
    int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    // int ** C = Encrypt(1, publicKey, lwe); // C[N, N]
    // int ** C2 = Encrypt(1, publicKey, lwe); // C[N, N]
    // unsigned char a = 255;
    // unsigned char b = 189;
    // cbyte AC = ByteEncrypt(b, publicKey, lwe); // C[N, N]
    // cbyte BC = ByteEncrypt(a, publicKey, lwe); // C[N, N]
    // cbyte OPERATION = ByteXOR(AC, BC, lwe); // C[N, N]
    // byte C4 = ByteDecrypt(AC, v, lwe); // C[N, N]
    // byte C5 = ByteDecrypt(BC, v, lwe); // C[N, N]
    // byte C6 = ByteDecrypt(OPERATION, v, lwe); // C[N, N]

    // int * bVector = GenerateBinaryVector(10);

    // int bitSize;
    // unsigned char * m = compressBitArray(bVector, 10, &bitSize);
    // int * h = decompressBitArray(m, 10);

    // printVector(bVector, 10, "bVector");
    // printBitVector(m, 2);
    // printVector(h, 10, "h");

    time_test(400, publicKey, lwe);

    // printf(" \n result: %d %d %d\n", C4, C5, C6);
}