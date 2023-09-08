
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
    
    int ** publicKey = PublicKeyGen(t, m); // pubK [m, K+1]

    int * check = MultiplyVectorxMatrix(secretKey, publicKey, m, K+1); // check must be equal to error as A.s = e

    printf("check \n");
    for(int h = 0; h < K+1; h++){
        printf("[%d]: %d \n ", h, mod(check[h], q));
    }

    

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