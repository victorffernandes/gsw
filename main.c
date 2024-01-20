#include "gsw.c"
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
    int ** AND = HomomorphicXOR(C, C2, lwe);
    int Cr = Decrypt(C, v, lwe);
    int C2r = Decrypt(C2, v, lwe);
    int r = Decrypt(AND, v, lwe);

    printf(" \n result: %d %d %d \n", Cr, C2r, r);
}