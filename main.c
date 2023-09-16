#include "gsw.c"
#include <time.h>

int main()
{
    srand(time(NULL));
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);

    int ** R = GenerateBinaryMatrix(N,m);
    
    int ** A = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** RA = (MultiplyMatrixxMatrix(R, A, N, m, m, K+1));

    int ** r = applyRows(RA, N, K+1, &BitDecomp); // r [N, N]

    int ** Identity = GenerateIdentity(N, N);

    int ** mIdentity = MultiplyMatrixEscalar(1, Identity, N,N);

    int ** sum = SumMatrixxMatrix(mIdentity, r, N,N);

    int ** C = applyRows(sum, N,N, &Flatten);

    printf("C \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < N; j++){
            printf("%d ", C[i][j]);
        }
        printf("]\n");
    }

    printf("sum \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < N; j++){
            printf("%d ", sum[i][j]);
        }
        printf("]\n");
    }

    printf("mIdentity \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < N; j++){
            printf("%d ", mIdentity[i][j]);
        }
        printf("]\n");
    }

    printf("R \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < m; j++){
            printf("%d ", R[i][j]);
        }
        printf("]\n");
    }

    printf("A \n");
    for(int i = 0; i < m; i++){
        printf("[%d][", i);
        for(int j = 0; j < K+1; j++){
            printf("%d ", A[i][j]);
        }
        printf("]\n");
    }

    printf("RA \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < K+1; j++){
            printf("%d ", (RA[i][j]));
        }
        printf("]\n");
    }    

    printf("r \n");
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < N; j++){
            printf("%d ", r[i][j]);
        }
        printf("]\n");
    }

    printf("message: %d \n", InternalProduct(C[L-1], v, L) / v[L-1]);

    int res[L];
    for(int j = L-1; j > 0; j--){
        int ip = InternalProduct(C[j], v, L);
        int dec = ip / v[j];
        res[j] = dec;
        printf("%d ", dec);
    }


    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}