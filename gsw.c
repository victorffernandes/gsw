#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define q 128 // q possui kappa bits (depende de lambda e L)
#define n 8 // dimensão do reticulado (depende de lambda e L)

int mod(int m, int l){
    if (m<0){
        return l + (m%l);
    }

    return m%l;
}

int ** GenerateG(int rows, int columns){
    int ** matrix = (int **)malloc(sizeof(int *) * rows);

    for (int i = 0; i< rows; i++){
        matrix[i] = (int *)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++){
            matrix[i][j] = 1/ powf(2, j);
        }
    }

    return matrix;
}

void printMatrix(int ** m, int r, int c, char * label){
    printf("\n %s \n", label);
    for(int i = 0; i< r; i++){
        for(int j = 0; j< c; j++){
            printf("%d ", m[i][j]);
        }
        printf("\n ");
    }
    printf("\n");
}

void printVector(int * v, int size, char * label){
    printf("\n %s ", label);
    for(int i = 0; i< size; i++){
        printf("%d ", v[i]);
    }
    printf("\n");
}

void printVectorf(float * v, int size, char * label){
    printf("\n %s ", label);
    for(int i = 0; i< size; i++){
        printf("%f ", v[i]);
    }
    printf("\n");
}

int ** applyRows(int ** matrix, int rows, int columns, int * (*f) (int * v, int size)){
    int ** result = (int **)malloc(sizeof(int * ) * rows);

    for (int j = 0; j < rows; j++){
        result[j] = (*f)(matrix[j], columns);
    }

    return result;
}

int * BitDecomp (int * vector, int L){
    int * result = (int *)malloc(sizeof(int) * L * n);

    for (int j = 0; j < n; j++){
        for (int i = 0; i < L; i++){
            result[j*L + i] = (vector[j] >> i) & 1;
        }
    }

    return result;
}

int * BitDecompInverse (int * bitVector, int LK){
    int * result = (int *)malloc(sizeof(int) * n);

    for (int j = 0; j < n; j++){
        int sum = 0;
        for (int i = 0; i < LK/n; i++){
            sum += (int)(pow(2,i)) * bitVector[(LK/n)*j + i];
        }
        result[j] = sum;
    }

    return result;
}

int * Flatten (int * bitVector, int LK){
    int * v =  BitDecompInverse(bitVector, LK);
    return BitDecomp(v, n);
}

int rand_ringz(){
    return (rand() % ((q))); // (rand() % (upper - lower + 1)) + (lower); // estava +1 
} 

int rand_error(){
    return (rand() % ((2))); // (rand() % (upper - lower + 1)) + (lower); // estava +1 
} 

int * Powersof2_(int * b, int size, int L){
    int * result = (int *)malloc(sizeof(int) * L * size);

    for(int j = 0; j < size; j++){
        for (int i = 0; i < L; i++){            
            int p = 1 << i; //  int p = (int) (pow(2, i));
            result[L*j + i] = b[j] * p;
        }
    }
    return result;
}

int * Powersof2(int * b, int k, int l){
    int * result = (int *)malloc(sizeof(int) * k * l);

    for (int i = 0; i < k; i++){            
        for(int j = 0; j < l; j++){
            int p = 1 << j; //  int p = (int) (pow(2, i));
            result[l*i + j] = b[i] * p;
        }
    }
    return result;
}


int * GenerateVector(int size){
    int * vector = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        vector[i] = rand_ringz();
        // printf("rand_ringz: %d \n", vector[i]);
    }

    return vector;
}

int * GenerateErrorVector(int size){
    int * vector = (int *)malloc(sizeof(int) * size);

    for (int i = 0; i< size; i++){
        vector[i] = rand_error();
    }

    return vector;
}

int InternalProduct(int * v1, int * v2, int size){
    int value = 0;

    for(int i = 0; i < size; i++){
        value += v1[i] * v2[i];
    }

    return value;
}

int * MultiplyVectorxVector(int * v1, int * v2, int size){
    int * result = (int *)malloc(sizeof(int  ) * size);

    for(int i = 0; i < size; i++){
        result[i] = v1[i] * v2[i];
    }

    return result;
}

float * DivideVectorxVector(int * v1, int * v2, float size){
    float * result = (float *)malloc(sizeof(float  ) * size);

    for(int i = 0; i < size; i++){
        result[i] = (float)v1[i] / (float)v2[i];
    }

    return result;
}

int ** MultiplyMatrixEscalar(int e, int ** matrix, int r, int c){
    int ** result = (int **)malloc(sizeof(int * ) * r);

    for (int i = 0; i< r; i++){
        result[i] = (int *)malloc(sizeof(int) * c);
        for (int j = 0; j < c; j++){
            result[i][j] = mod(matrix[i][j] * e, q);
        }
    }

    return result;
}

int * MultiplyVectorxMatrix(int * v, int ** matrix, int r, int c){
    int * result = (int *)malloc(sizeof(int) * r);

    for (int i = 0; i< r; i++){
        for (int j = 0; j< c; j++){
            result[i] += v[j] * matrix[i][j];
        }
        result[i] = result[i];
    }
    return result;
}

int * MultiplyVectorxMatrixOverQ(int * v, int ** matrix, int r, int c){
    int * result = (int *)malloc(sizeof(int) * r);

    for (int i = 0; i< r; i++){
        for (int j = 0; j< c; j++){
            result[i] += v[j] * matrix[i][j];
        }
        result[i] = mod(result[i], q);
    }
    return result;
}

int ** SumMatrixxMatrix(int ** m1, int ** m2, int r, int c){
    int ** result = (int **)malloc(sizeof(int * ) * r);

    for (int i = 0; i< r; i++){
        result[i] = (int *)malloc(sizeof(int) * c);
        for (int j = 0; j < c; j++){
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return result;
}

int ** MultiplyMatrixxMatrix(int ** m1, int ** m2, int r1, int c1, int r2, int c2){
    int ** result = (int **)malloc(sizeof(int * ) * r1);

    for (int i = 0; i< r1; i++){
        result[i] = (int *)malloc(sizeof(int) * c2);
        for (int j = 0; j < c2; j++){
            int sum = 0;
            for (int z = 0; z < c1; z++){
                sum += (m1[i][z]) * (m2[z][j]);
            }

            result[i][j] = mod(sum,q);
        }
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
            matrix[i][j] =  rand() % 2 ;// TODO: get int by parameter q
        }
    }

    return matrix;
}


int ** PublicKeyGen(int * t, int m){
    int * error = GenerateErrorVector(m);
    // printVector(error, m, "error vector");
    int ** B = GenerateMatrix(m,n);
    // printMatrix(B, m,n , "B ");
    int * b = MultiplyVectorxMatrixOverQ(t, B, m, n);

    // printVector(b, m, "b ");

    int ** A = (int **)malloc(sizeof(int *) * (m));
    // set first column of A with b
    for(int k = 0; k < m; k++){
        A[k] = (int *)malloc(sizeof(int *) * (n + 1));
        A[k][0] = b[k];
    }

    // set A with B (except first column)
    for(int i = 0; i < m; i++){
        for(int j = 1; j < n+1; j++){
            A[i][j] = B[i][j-1];
        }
    }

    // printMatrix(A, m, n+1, "A ");

    // TODO: must free B, b and error
    return A;
}


int * SecretKeyGen(int * t){
    int * sk = (int *)malloc(sizeof(int) * (n + 1));
    for (int i = 1; i < n+1; i++){
        sk[i] = q - t[i-1]; // antes era - t[i-1]
    }

    sk[0] = 1;

    return sk;
}


int ** Encrypt(int message, int ** pubKey, int m, int N){
    // BitDecomp (R * A)
    int ** R = GenerateBinaryMatrix(N,m);
    int ** RA = (MultiplyMatrixxMatrix(R, pubKey, N, m, m, n+1));
    int ** r = applyRows(RA, N, n+1, &BitDecomp); // r [N, N]

    // m * In
    int ** Identity = GenerateIdentity(N, N);
    int ** mIdentity = MultiplyMatrixEscalar(message, Identity, N,N);

    // m*In + BitDecomp(R * A)
    int ** sum = SumMatrixxMatrix(mIdentity, r, N,N);
    
    // Flatten (m*In + BitDecomp(R * A))
    int ** C = sum; //applyRows(sum, N,N, &Flatten);

    //TODO: free

    return C;
}


int Decrypt(int ** C, int * v, int L){
    int ith = log2(q);
    float message =  mod((InternalProduct(C[ith], v, L)) / v[ith], q);
    printf("message: %f \n", message);
    printf("internal [%d]: %d  v[%d]: %d \n",ith,  mod(InternalProduct(C[ith], v, L), q ), ith, v[ith]);
    return message;
}