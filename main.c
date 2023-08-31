
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define q 128
#define K 8


int * BitDecomp (int * vector, int L){
    int * result = (int *)malloc(sizeof(int) * L * K);

    for (int j = 0; j < K; j++){
        for (int i = 0; i < L; i++){
            result[j*L + i] = (vector[j] >> i) & 1;
        }
    }

    return result;
}

int * BitDecompInverse (int * bitVector, int L){
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
    int * v =  BitDecompInverse(bitVector, L);
    return BitDecomp(v, L);
}

int rand_ringz(){
    return (rand() % ((2 * q) + 1)) - q;
} 

int * Powersof2(int * vector, int L){
    int * result = (int *)malloc(sizeof(int) * L * K);

    for(int j = 0; j < K; j++){
        for (int i = 0; i < L; i++){
            int p = (int) (pow(2, i));
            result[L*j + i] = vector[j] * p;
        }
    }
    return result;
}

int * GenerateVector(int size){
    int * vector = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        vector[i] = rand_ringz();
    }

    return vector;
}

int ** GenerateMatrix(int  rows, int columns){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for (int i = 0; i< rows; i++){
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++){
            matrix[i][j] = rand_ringz();
        }
    }

    return matrix;
}

int * SecretKeyGen(){
    int * s = GenerateVector( K + 1);
    s[0] = 1;

    return s;
}

int main()
{
    srand(4);

    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * secretKey = SecretKeyGen();
    int * v = Powersof2(secretKey, L);

    for(int h = 0; h <= K; h++){
        printf("secret key[%d]: %d \n ", h, secretKey[h]);
    }

    for(int h = 0; h < K * L; h++){
        printf("v[%d]: %d \n ", h, v[h]);
    }

    // for(int h = 0; h < N; h++){
    //     printf("v[%d]: %d \n ", h, v[h]);
    // }

    printf("q: %d, K: %d, L: %d, N: %d", q, K, L, N);
}