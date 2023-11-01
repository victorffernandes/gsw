#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calc.c"


typedef  struct lwe_instance {
    int q; // q possui kappa bits (depende de lambda e l)
    int n; // dimensão do reticulado (depende de lambda e l)
    int N; // (n+1) * l
    int l; // l bits of q
    int m; // arbitrary parameter  (depende de lambda e l)
    int lambda; // security parameter
} lwe_instance;

lwe_instance GenerateLweInstance(int lambda){
    lwe_instance * l = (lwe_instance *) malloc(sizeof(lwe_instance));
    l->lambda = lambda;
    l->n = 16;
    l->q = 1 << lambda;
    l->l = log2(l->q);
    l->N = (l->n + 1) * l->l;
    l->m = l->n * l->l;

    printf("q: %d, n: %d, l: %d, N: %d m: %d", l->q, l->n, l->l, l->N, l->m);
    return *l;
}


int Decrypt(int ** C, int * v, lwe_instance lwe){
    float message =  mod((InternalProduct(C[lwe.l-1], v, lwe.N)), lwe.q)/ v[lwe.l-1];
    return (int) message;
}


int ** GenerateG(lwe_instance lwe){
    int ** matrix = (int **)malloc(sizeof(int *) * lwe.n);

    for (int i = 0; i< lwe.N; i++){
        matrix[i] = (int *)malloc(sizeof(int) * lwe.n+1);
        for (int j = 0; j < lwe.n + 1; j++){
             //  int p = (int) (pow(2, i));
             matrix[i][j] = 0;
            if (j > (i)*lwe.l && j < (i+1)*lwe.l ){
                int p = 1 << j;
                matrix[i][j] = p;
            }
        }
    }

    return matrix;
}


int ** applyRows(int ** matrix, int rows, int size, int * (*f) (int * v, int s, lwe_instance l), lwe_instance lwe){
    int ** result = (int **)malloc(sizeof(int * ) * rows);

    for (int j = 0; j < rows; j++){
        result[j] = (*f)(matrix[j], size, lwe);
    }

    return result;
}

int * BitDecomp (int * vector, int size, lwe_instance lwe){
    int * result = (int *)malloc(sizeof(int) * size * lwe.l);

    for (int j = 0; j < size; j++){
        for (int i = 0; i < lwe.l; i++){
            result[j*lwe.l + i] = (vector[j] >> i) & 1;
        }
    }

    return result;
}

int * BitDecompInverse (int * bitVector, int size, lwe_instance lwe){
    int inverseSize = (size/lwe.l);
    int * result = (int *)malloc(sizeof(int) * (size/lwe.l));

    for (int j = 0; j < inverseSize; j++){
        int sum = 0;
        for (int i = 0; i < lwe.l; i++){
            sum += (1 << i) * bitVector[(lwe.l)*j + i];
        }
        result[j] = mod(sum, lwe.q);
    }

    return result;
}

int * Flatten (int * bitVector, int size, lwe_instance lwe){
    int * v =  BitDecompInverse(bitVector, size, lwe);
    return BitDecomp(v, size/lwe.l, lwe);
}

int rand_ringz(int q){
    return (rand() % ((q))); // (rand() % (upper - lower + 1)) + (lower); // estava +1 
} 

int rand_error(){
    return (rand() % ((2))); // (rand() % (upper - lower + 1)) + (lower); // estava +1 
} 

int * Powersof2(int * b, lwe_instance lwe){
    int * result = (int *)malloc(sizeof(int) * (lwe.l) * (lwe.n + 1));

    for(int j = 0; j < (lwe.n + 1); j++){
        for (int i = 0; i < lwe.l; i++){            
            int p = 1 << i; //  int p = (int) (pow(2, i));
            result[lwe.l*j + i] = b[j] * p;
        }
    }
    return result;
}

int * Powersof2_(int * b, int k, int l){
    int * result = (int *)malloc(sizeof(int) * k * l);

    for (int i = 0; i < k; i++){            
        for(int j = 0; j < l; j++){
            int p = 1 << j; //  int p = (int) (pow(2, i));
            result[l*i + j] = b[i] * p;
        }
    }
    return result;
}


int * GenerateVector(int size, lwe_instance lwe){
    int * vector = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        vector[i] = rand_ringz(lwe.q);
        // printf("rand_ringz: %d \n", vector[i]);
    }

    return vector;
}



int ** PublicKeyGen(int * t, lwe_instance lwe){
    int * error = GenerateErrorVector(lwe.m);
    printVector(error, lwe.m, "error");
    int ** B = GenerateMatrixOverQ(lwe.m,lwe.n, lwe.q);
    // printMatrix(B, m,n , "B ");
    int * b = SumVectorOverQ(MultiplyVectorxMatrixOverQ(t, B, lwe.m, lwe.n, lwe.q), error, lwe.m, lwe.q);

    int ** A = (int **)malloc(sizeof(int *) * (lwe.m));
    // set first column of A with b
    for(int k = 0; k < lwe.m; k++){
        A[k] = (int *)malloc(sizeof(int *) * (lwe.n + 1));
        A[k][0] = b[k];
    }

    // set A with B (except first column)
    for(int i = 0; i < lwe.m; i++){
        for(int j = 1; j < lwe.n+1; j++){
            A[i][j] = B[i][j-1];
        }
    }

    // TODO: must free B, b and error
    return A; // [m, n+1]
}


int * SecretKeyGen(int * t, lwe_instance lwe){
    int * sk = (int *)malloc(sizeof(int) * (lwe.n + 1));
    for (int i = 1; i < lwe.n+1; i++){
        sk[i] = lwe.q - t[i-1]; // antes era - t[i-1]
    }

    sk[0] = 1;

    return sk;
}

int ** Encrypt(int message, int ** pubKey, lwe_instance lwe){
    // BitDecomp (R * A)
    int ** R = GenerateBinaryMatrix(lwe.N,lwe.m); // [N, m]
    int ** RA = MultiplyMatrixxMatrixOverQ(R, pubKey, lwe.N, lwe.m, lwe.m, lwe.n+1, lwe.q); // [N, n+1]

    int ** BitDecomRA = applyRows(RA, lwe.N, lwe.n+1, &BitDecomp, lwe); // r [N, N]

    // // m * In
    int ** Identity = GenerateIdentity(lwe.N, lwe.N);
    int ** mIdentity = MultiplyMatrixEscalarOverQ(message, Identity, lwe.N, lwe.N, lwe.q); // [N, N]

    // printMatrix(mIdentity, lwe.N, lwe.N, "Identity");

    // m*In + BitDecomp(R * A)
    int ** sum = SumMatrixxMatrix(mIdentity, BitDecomRA, lwe.N, lwe.N); // [N, N]

    // Flatten (m*In + BitDecomp(R * A)) 
    int ** C = applyRows(sum, lwe.N, lwe.N, &Flatten, lwe);

    return C;
}
