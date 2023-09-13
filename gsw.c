#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define q 128
#define K 8

int mod(int m, int l){
    if (m<0){
        return l + (m%l);
    }

    return m%l;
}

int ** applyRows(int ** matrix, int rows, int columns, int * (*f) (int * v, int n)){
    int ** result = (int **)malloc(sizeof(int) * rows);

    for (int j = 0; j < rows; j++){
        result[j] = (*f)(matrix[j], columns);
    }

    return result;
}

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
    return (rand() % ((q) + 1)); // (rand() % (upper - lower + 1)) + (lower);
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

int * GenerateErrorVector(int size){
    int * vector = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        vector[i] = (rand() % (5)) - 2;
    }

    return vector;
}

int * MultiplyVectorxMatrix(int * v, int ** matrix, int r, int c){
    int * result = (int *)malloc(sizeof(int) * r);

    for (int i = 0; i< r; i++){
        for (int j = 0; j< c; j++){
            result[i] += v[j] * matrix[i][j];
        }
        result[i] = result[i] % q;
    }
    return result;
}

int * SumVector(int * v1, int * v2, int size){
    int * result = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        result[i] = (v1[i] + v2[i]) % q;
    }

    return result;
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

int ** PublicKeyGen(int * t, int m){
    int * error = GenerateErrorVector(m);
    int ** B = GenerateMatrix(m,K);
    int * b = SumVector(MultiplyVectorxMatrix(t, B, m, K), error, m);

    int ** A = (int **)malloc(sizeof(int *) * (m));
    // set first column of A with b
    for(int k = 0; k < m; k++){
        A[k] = (int *)malloc(sizeof(int *) * (K + 1));
        A[k][0] = b[k];
    }

    // set A with B (except first column)
    for(int i = 0; i < m; i++){
        for(int j = 1; j < K+1; j++){
            A[i][j] = B[i][j-1];
        }
    }

    // TODO: must free B, b and error
    return A;
}


int * SecretKeyGen(int * t){
    int * sk = (int *)malloc(sizeof(int) * (K + 1));
    for (int i = 1; i < K+1; i++){
        sk[i] = -t[i-1];
    }

    sk[0] = 1;

    return sk;
}

int ** GenerateIdentity(int rows, int columns){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for (int i = 0; i< rows; i++){
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++){
            if(i == j){
                matrix[i][j] = 1;
            } else{
                matrix[i][j] = 0;
            }
        }
    }

    return matrix;
}

int ** GenerateBinaryMatrix(int rows, int columns){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for (int i = 0; i< rows; i++){
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++){
            matrix[i][j] =  rand()/  1000000000 ;
        }
    }

    return matrix;
}
