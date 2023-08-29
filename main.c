
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define q 101
#define K 20


int * BitDecomp (int * vector, int L){
    int * result = (int *)malloc(sizeof(int) * L * K);

    for (int j = 0; j < K; j++){
        for (int i = 0; i < L; i++){
            result[j*L + i] = (vector[j] >> i) & 1;
        }
    }

    return result;
}

int * InverseBitDecomp (int * bitVector, int L){
    int * result = (int *)malloc(sizeof(int) * L * K);

    for (int j = 0; j < K; j++){
        int sum = 0;
        for (int i = 0; i < L; i++){
            sum += (int)(pow(2,i)) * bitVector[L*j + i];
        }
        result[j] = sum;
    }

    return result;
}

int * Flatten (int * bitVector, int L){
    int * v =  InverseBitDecomp(bitVector, L);
    return BitDecomp(v, L);
}

int main()
{

    int L = log2(q) + 1;
    int N = K * L;

    int v[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};

    int * r = BitDecomp(v, L);
    int * m = InverseBitDecomp(r, L);

    for(int h = 0; h < K; h++){
        printf("InverseBitdecomp[%d]: %d \n ", h, m[h]);
    }




    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}