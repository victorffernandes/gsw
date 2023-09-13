
#include "gsw.c"

int main()
{
    srand(4);

    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);

    int ** binm = GenerateBinaryMatrix(m,K+1);
        for(int i = 0; i < m; i++){
        printf("BM: [%d][", i);
        for(int j = 0; j < K+1; j++){
            printf("%d ", binm[i][j]);
        }
        printf("]\n");
    }
    
    int ** publicKey = PublicKeyGen(t, m); // pubK [m, K+1]

    int ** r = applyRows(publicKey, m, K+1, &BitDecomp);


    printf("public key \n");
    for(int i = 0; i < m; i++){
        printf("[%d][", i);
        for(int j = 0; j < K+1; j++){
            printf("%d ", publicKey[i][j]);
        }
        printf("]\n");
    }
    printf("secret key \n");
    for(int h = 0; h < K+1; h++){
        printf("[%d]: %d \n ", h, secretKey[h]);
    }

    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}