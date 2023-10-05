#include "gsw.c"
#include <time.h>
#include <math.h>



int main()
{
    srand(0);
    int k = log2(q) + 1;
    int l = log2(q) + 1;
    int N = (n+1) * l;
    int m = N;

    int * t = GenerateVector(n);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, l);
    int ** A = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** C = Encrypt(0, A, m, N); // C[N, N]
    // int ** Cs = MultiplyMatrixEscalar(1, C, N, N);
    // int ** r = applyRows(C, N, N, &BitDecompInverse); // r [N, N]

    int message = Decrypt(C, v, l);
    // int res[L-2];


    // for (int i = 1; i < N; i++){
    //      for (int j = 1; j < K; j++){
    //         printf("%d ",r[i][j] );
    //      }
    //      printf("\n");
    // }



    printf("q: %d, n: %d, l: %d, N: %d", q, n, l, N);
}