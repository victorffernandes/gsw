#include "gsw.c"
#include <time.h>
#include <math.h>



int main()
{
    srand(time(NULL));
    int L = log2(q) + 1;
    int N = K * L;
    int m = N;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);
    int * v_ = MultiplyVectorEscalar(1, v, N);
    int ** A = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** C = Encrypt(0, A, m, N); // C[N, N]
    int * Cv = MultiplyVectorxMatrix(v, C, N, N);

    int * msg = MultiplyVectorxVector(C[0], v, N);
    // int ** Cs = MultiplyMatrixEscalar(1, C, N, N);
    int ** CBitDecompInverse = applyRows(C, N, N, &BitDecompInverse); // r [N, N]

    int message = Decrypt(C, v, L);
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

    printf(" i: %d \n",L-2);
    printf("v[l-2]: %d  \n",v[L-1]); 
    printf("Cv[l-2]: %d  \n", Cv[L-2]/ v[L-1]);   
    printf("msg[l-2]: %d  \n", msg[L-2]);   

    printf("v:  \n"); 
    for (int i = 0; i < K; i++){
        for (int j = 0; j < L; j++){
            printf("%d ", v[i * K + j]);
        }
        printf("\n");
    }

    printf("C[0]:  \n"); 
    for (int i = 0; i < K; i++){
        for (int j = 0; j < L; j++){
            printf("%d ", C[0][i * K + j]);
        }
        printf("\n");
    }

    printf("msg:  \n"); 
    for (int i = 0; i < K; i++){
        int s = 0;
        for (int j = 0; j < L; j++){
            s += msg[i * K + j];
            printf("%d ", msg[i * K + j]);
        }
        printf(" |%d \n", mod(s,q));
    }

    printf("CBitDecompInverse:  \n"); 
    for (int i = 0; i < N; i++){
        int s = 0;
        for (int j = 0; j < N; j++){
            printf("%d ", CBitDecompInverse[i][j]);
        }
        printf(" | \n");
    }




    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}