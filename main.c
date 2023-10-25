#include "gsw.c"
#include <time.h>
#include <math.h>



int main()
{
    // setup (a partir de q, n e distribuição de erro Chi)
    srand(4);
    int k = log2(q) + 1;
    int l = log2(q) + 1;
    int N = (n+1) * l;
    int m = N;  // m = O(n * log q)

    int * t = GenerateVector(n);
    int * t_f = GenerateVector(n);
    int * secretKey = SecretKeyGen(t);
    int * secretKey_f = SecretKeyGen(t_f);
    int * v = Powersof2(secretKey, n+1, l);
    int * v_f = Powersof2(secretKey_f, n+1, l);
    int ** A = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** C = Encrypt(62, A, m, N); // C[N, N]
    int * Cv = MultiplyVectorxMatrix(v_f, C, N, N);

    // int * msg = MultiplyVectorxMatrixOverQ(v, C, N, N);
    // int p = InternalProduct(v_f, C[l-2], N);
    // printf("valor:: %d \n", mod(p /v[l-2] , q));
    // int ** Cs = MultiplyMatrixEscalar(1, C, N, N);
    // int ** CBitDecompInverse = applyRows(C, N, N, &BitDecompInverse); // r [N, N]

    int message = Decrypt(C, v_f, l);
    // int res[L-2];

    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < N; j++){
    //         printf("%d ", C[i][j]);
    //     }
    // }
    // printf("\n");
    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < K; j++){
    //         printf("%d ", r[i][j]);
    //     }
    // }

    // printf(" i: %d \n",l-2);
    // printf("v[l-2]: %d  \n",v[l-1]); 
    // printf("Cv[l-2]: %d  \n", Cv[l-2]/ v[l-1]);
    printVector(v, N, "v: ");
    printVector(v_f, N, "v_f: ");


    // printf("v:  \n"); 
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < l; j++){
    //         printf("%d ", v[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // printf("C[0]:  \n"); 
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < l; j++){
    //         printf("%d ", C[0][i * n + j]);
    //     }
    //     printf("\n");
    // }

    // printf("msg:  \n"); 
    // for (int i = 0; i < n; i++){
    //     int s = 0;
    //     for (int j = 0; j < l; j++){
    //         s += msg[i * n + j];
    //         printf("%d ", msg[i * n + j]);
    //     }
    //     printf(" |%d \n", mod(s,q));
    // }

    // printf("CBitDecompInverse:  \n"); 
    // for (int i = 0; i < N; i++){
    //     int s = 0;
    //     for (int j = 0; j < N; j++){
    //         printf("%d ", CBitDecompInverse[i][j]);
    //     }
    //     printf(" | \n");
    // }




    printf("q: %d, n: %d, l: %d, N: %d", q, n, l, N);
}