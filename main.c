#include "gsw.c"
#include <time.h>


int ** Encrypt(int message, int ** pubKey, int m, int N){
    // BitDecomp (R * A)
    int ** R = GenerateBinaryMatrix(N,m);
    int ** RA = (MultiplyMatrixxMatrix(R, pubKey, N, m, m, K+1));
    int ** r = applyRows(RA, N, K+1, &BitDecomp); // r [N, N]

    // m * In
    int ** Identity = GenerateIdentity(N, N);
    int ** mIdentity = MultiplyMatrixEscalar(message, Identity, N,N);

    // m*In + BitDecomp(R * A)
    int ** sum = SumMatrixxMatrix(mIdentity, r, N,N);

    // Flatten (m*In + BitDecomp(R * A))
    int ** C = applyRows(sum, N,N, &Flatten);

    printf("C \n");
    int ** C_f = applyRows(sum, N,N, &BitDecompInverse);
    for(int i = 0; i < N; i++){
        printf("[%d][", i);
        for(int j = 0; j < K; j++){
            printf("%d ", mod(C_f[i][j], q));
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

    // printf("R \n");
    // for(int i = 0; i < N; i++){
    //     printf("[%d][", i);
    //     for(int j = 0; j < m; j++){
    //         printf("%d ", R[i][j]);
    //     }
    //     printf("]\n");
    // }

    // printf("A \n");
    // for(int i = 0; i < m; i++){
    //     printf("[%d][", i);
    //     for(int j = 0; j < K+1; j++){
    //         printf("%d ", A[i][j]);
    //     }
    //     printf("]\n");
    // }

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

    return C;
}

int Decrypt(int ** C, int * v, int L){
    int ith = L-1; // L-1
    int message =  mod(InternalProduct(C[ith], v, L), q )/ v[ith];
    printf("message: %d \n", message);
    printf("internal [%d]: %d  v[%d]: %d \n",ith,  mod(InternalProduct(C[ith], v, L), q ), ith, v[ith]);
    return message;
}

int main()
{
    srand(time(NULL));
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);
    int ** A = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** C = Encrypt(1, A, m, N);

    int message = Decrypt(C, v, L);

    int res[L];
    for(int j = L; j > 0; j--){
        int ip = InternalProduct(C[j], v, j);
        int dec = ip / v[j-1];
        res[j] = dec;
        
        printf("%d ", dec);
    }


    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}