#include "lib/gsw.c"
#include <time.h>
#include <math.h>



int main()
{
    // setup (a partir de q, n e distribuição de erro Chi)
    srand(4);

    lwe_instance lwe = GenerateLweInstance(25);

    int * t = GenerateVector(lwe.n, lwe);
    int * secretKey = SecretKeyGen(t, lwe);
    int * v = Powersof2(secretKey, lwe);
    
    int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    int ** C = Encrypt(1, publicKey, lwe); // C[N, N]
    int ** C2 = Encrypt(1, publicKey, lwe); // C[N, N]
    unsigned char a = 0;
    unsigned char b = 1;
    cbyte C3 = ByteEncrypt(b, publicKey, lwe); // C[N, N]
    cbyte C3_ = ByteEncrypt(a, publicKey, lwe); // C[N, N]
    cbyte C3__ = ByteXOR(C3, C3_, v, lwe); // C[N, N]
    byte C4 = ByteDecrypt(C3__, v, lwe); // C[N, N]
    byte C5 = ByteDecrypt(C3_, v, lwe); // C[N, N]
    byte C6 = ByteDecrypt(C3, v, lwe); // C[N, N]

    printf(" \n result: %d %d %d\n", C5, C6, C4);
}