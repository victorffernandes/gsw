#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "calc.c"

const int BYTE_LENGTH = 8;

typedef int*** cbyte;
typedef int** cbit;
typedef unsigned char byte;

typedef struct lwe_instance
{
    int q;      // q possui kappa bits (depende de lambda e l)
    int n;      // dimensão do reticulado (depende de lambda e l)
    int N;      // (n+1) * l
    int l;      // l bits of q
    int m;      // arbitrary parameter  (depende de lambda e l)
    int B;      // B bounded error
    int lambda; // security parameter
} lwe_instance;

lwe_instance GenerateLweInstance(int lambda)
{
    srand(time(NULL));
    lwe_instance* l = (lwe_instance*)malloc(sizeof(lwe_instance));
    l->lambda = lambda;
    l->n = lambda;
    l->q = 1 << lambda;
    l->l = lambda;
    l->N = (l->n + 1) * l->l;
    l->m = 2 * l->n * l->l + 1;
    l->B = 2;

    printf("q: %d, n: %d, l: %d, N: %d m: %d", l->q, l->n, l->l, l->N, l->m);
    // printf("\n q/B: %d, L: 4,  8 (N+1)^L: %d ", l->q / l->B, 8 * pow(l->N + 1, 4));
    return *l;
}

int Decrypt(int** C, int* v, lwe_instance lwe)
{
    float message = mod((InternalProduct(C[lwe.l - 1], v, lwe.N)), lwe.q) / v[lwe.l - 1];
    return (int)message;
}

int MPDecrypt(int** C, int* v, lwe_instance lwe)
{
    int* checkSUM = MultiplyVectorxMatrixOverQ(v, C, lwe.N, lwe.N, lwe.q); // [N]

    int value = 0;
    for (int i = 0; i < lwe.l - 2; i++)
    {
        int p = 1 << i;
        value += (1 << (lwe.l - 2 - i)) * (((checkSUM[i] / p) >> (lwe.l - 2 - i)) & 1);
    }
    printf(" \n estimated value: %d", value);
    return value;
}

int** GenerateG(lwe_instance lwe)
{
    int** matrix = (int**)malloc(sizeof(int*) * lwe.n);

    for (int i = 0; i < lwe.N; i++)
    {
        matrix[i] = (int*)malloc(sizeof(int) * lwe.n + 1);
        for (int j = 0; j < lwe.n + 1; j++)
        {
            //  int p = (int) (pow(2, i));
            matrix[i][j] = 0;
            if (j > (i)*lwe.l && j < (i + 1) * lwe.l)
            {
                int p = 1 << j;
                matrix[i][j] = p;
            }
        }
    }

    return matrix;
}

int** applyRows(int** matrix, int rows, int size, int* (*f)(int* v, int s, lwe_instance l), lwe_instance lwe)
{
    int** result = (int**)malloc(sizeof(int*) * rows);

    for (int j = 0; j < rows; j++)
    {
        result[j] = (*f)(matrix[j], size, lwe);
    }

    return result;
}

int* BitDecomp(int* vector, int size, lwe_instance lwe)
{
    int* result = (int*)malloc(sizeof(int) * size * lwe.l);

    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < lwe.l; i++)
        {
            result[j * lwe.l + i] = (vector[j] >> i) & 1;
        }
    }

    return result;
}

int* BitDecompInverse(int* bitVector, int size, lwe_instance lwe)
{
    int inverseSize = (size / lwe.l);
    int* result = (int*)malloc(sizeof(int) * (size / lwe.l));

    for (int j = 0; j < inverseSize; j++)
    {
        int sum = 0;
        for (int i = 0; i < lwe.l; i++)
        {
            sum += (1 << i) * bitVector[(lwe.l) * j + i];
        }
        result[j] = mod(sum, lwe.q);
    }

    return result;
}

int* Flatten(int* bitVector, int size, lwe_instance lwe)
{
    int* v = BitDecompInverse(bitVector, size, lwe);
    return BitDecomp(v, size / lwe.l, lwe);
}

int* Powersof2(int* b, lwe_instance lwe)
{
    int* result = (int*)malloc(sizeof(int) * lwe.N);

    for (int j = 0; j < (lwe.n + 1); j++)
    {
        for (int i = 0; i < lwe.l; i++)
        {
            int p = 1 << i; //  int p = (int) (pow(2, i));
            result[lwe.l * j + i] = b[j] * p;
        }
    }
    return result;
}

int* Powersof2_(int* b, int k, int l)
{
    int* result = (int*)malloc(sizeof(int) * k * l);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < l; j++)
        {
            int p = 1 << j; //  int p = (int) (pow(2, i));
            result[l * i + j] = b[i] * p;
        }
    }
    return result;
}

int* GenerateVector(int size, lwe_instance lwe)
{
    int* vector = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++)
    {
        vector[i] = rand_ringz(lwe.q);
        // printf("rand_ringz: %d \n", vector[i]);
    }

    return vector;
}

int** PublicKeyGen(int* t, lwe_instance lwe)
{
    int* error = GenerateErrorVector(lwe.m, lwe.B);
    int** B = GenerateMatrixOverQ(lwe.m, lwe.n, lwe.q);
    int* b = SumVectorOverQ(MultiplyVectorxMatrixOverQ(t, B, lwe.m, lwe.n, lwe.q), error, lwe.m, lwe.q);

    int** A = (int**)malloc(sizeof(int*) * (lwe.m));
    // set first column of A with b
    for (int k = 0; k < lwe.m; k++)
    {
        A[k] = (int*)malloc(sizeof(int*) * (lwe.n + 1));
        A[k][0] = b[k];
    }

    // set A with B (except first column)
    for (int i = 0; i < lwe.m; i++)
    {
        for (int j = 1; j < lwe.n + 1; j++)
        {
            A[i][j] = B[i][j - 1];
        }
    }

    // TODO: must free B, b and error
    return A; // [m, n+1]
}

int* SecretKeyGen(int* t, lwe_instance lwe)
{
    int* sk = (int*)malloc(sizeof(int) * (lwe.n + 1));
    for (int i = 1; i < lwe.n + 1; i++)
    {
        sk[i] = lwe.q - t[i - 1]; // antes era - t[i-1]
    }

    sk[0] = 1;

    return sk;
}

int** Encrypt(int  message, int** pubKey, lwe_instance lwe)
{
    // BitDecomp (R * A)
    int** R = GenerateBinaryMatrix(lwe.N, lwe.m);                                            // [N, m]
    int** RA = MultiplyMatrixxMatrixOverQ(R, pubKey, lwe.N, lwe.m, lwe.m, lwe.n + 1, lwe.q); // [N, n+1]

    int** BitDecomRA = applyRows(RA, lwe.N, lwe.n + 1, &BitDecomp, lwe); // r [N, N]

    // // m * In
    int** Identity = GenerateIdentity(lwe.N, lwe.N);
    int** mIdentity = MultiplyMatrixEscalarOverQ(message, Identity, lwe.N, lwe.N, lwe.q); // [N, N]

    // m*In + BitDecomp(R * A)
    int** sum = SumMatrixxMatrix(mIdentity, BitDecomRA, lwe.N, lwe.N); // [N, N]

    // Flatten (m*In + BitDecomp(R * A))
    int** C = applyRows(sum, lwe.N, lwe.N, &Flatten, lwe);

    return C;
}

int** HomomorphicSum(int** C1, int** C2, lwe_instance lwe)
{
    int** C3 = SumMatrixxMatrix(C1, C2, lwe.N, lwe.N);
    return applyRows(C3, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicMult(int** C1, int** C2, lwe_instance lwe)
{
    int** C3 = MultiplyMatrixxMatrixOverQ(C1, C2, lwe.N, lwe.N, lwe.N, lwe.N, lwe.q);
    return applyRows(C3, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicMultByConst(int** C1, int a, lwe_instance lwe)
{
    int** mIdentity = GenerateIdentityMultipliedByConst(lwe.N, lwe.N, a);

    // Ma = Flatten (In * a)
    int** Ma = applyRows(mIdentity, lwe.N, lwe.N, &Flatten, lwe);
    int** C3 = MultiplyMatrixxMatrixOverQ(C1, Ma, lwe.N, lwe.N, lwe.N, lwe.N, lwe.q);
    // Flatten (Ma * C)
    return applyRows(C3, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicAND(int** C1, int** C2, lwe_instance lwe)
{
    // C3 = C1 * C2
    int** C3 = MultiplyMatrixxMatrixOverQ(C1, C2, lwe.N, lwe.N, lwe.N, lwe.N, lwe.q);
    return applyRows(C3, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicNOT(int** C1, lwe_instance lwe)
{
    int** Identity = GenerateIdentity(lwe.N, lwe.N);

    // C4 = In - C1
    int** C4 = SubMatrixxMatrix(Identity, C1, lwe.N, lwe.N);
    // Flatten (C4)
    return applyRows(C4, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicNAND(int** C1, int** C2, int** Identity, lwe_instance lwe)
{
    // C3 = C1 * C2
    int** C3 = MultiplyMatrixxMatrixOverQ(C1, C2, lwe.N, lwe.N, lwe.N, lwe.N, lwe.q);
    // C4 = C3 - In
    int** C4 = SubMatrixxMatrix(Identity, C3, lwe.N, lwe.N);
    // Flatten (C1 * C2 - In)
    return applyRows(C4, lwe.N, lwe.N, &Flatten, lwe);
}

int** HomomorphicXOR(int** C1, int** C2, lwe_instance lwe)
{
    return HomomorphicSum(C1, C2, lwe);
}

cbyte ByteEncrypt(byte b, int** pubKey, lwe_instance lwe)
{
    int*** c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);
    for (int j = 0; j < BYTE_LENGTH; j++)
    {
        int p = (b >> j) & 1; //  int p = (int) (pow(2, i));
        c[j] = Encrypt(p, pubKey, lwe);
    }

    return c;
}

uint8_t ByteDecrypt(cbyte b, int* v, lwe_instance lwe)
{
    uint8_t* c = (uint8_t*)malloc(sizeof(unsigned char));

    for (int j = 0; j < BYTE_LENGTH; j++)
    {
        int p = Decrypt(b[j], v, lwe);
        *c = (*c & ~((unsigned char)1 << j)) | ((unsigned char)p << j);
    }

    return *c;
}

cbyte ByteAND(cbyte a, cbyte b, lwe_instance lwe)
{
    cbyte c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);
    for (int j = 0; j < BYTE_LENGTH; j++)
    {
        c[j] = HomomorphicAND(a[j], b[j], lwe);
    }

    return c;
}

cbyte ByteXOR(cbyte a, cbyte b, lwe_instance lwe)
{
    cbyte c = (int***)malloc(sizeof(int**) * BYTE_LENGTH);
    for (int j = 0; j < BYTE_LENGTH; j++)
    {
        c[j] = HomomorphicXOR(a[j], b[j], lwe);
    }

    return c;
}

void printBitVector(uint8_t* v, int size)
{
    printf("Bit Vector:");
    for (int n = 0; n < size; n++)
    {
        for (int j = 0; j < BYTE_LENGTH; j++)
        {
            int p = (v[n] >> j) & 1; //  int p = (int) (pow(2, i));
            printf("%d ", p);
        }
    }
    printf("\n");
}

void setBit(uint8_t* byte, int pos, int value)
{
    if (value)
        *byte |= (1 << pos);
    else
        *byte &= ~(1 << pos);
}

void setBitInteger(int* byte, int pos, int value)
{
    if (value)
        *byte |= (1 << pos);
    else
        *byte &= ~(1 << pos);
}

uint8_t* compressBitArray(int* array, int intSize, int* bitSize)
{
    *bitSize = (int)ceil(((double)intSize / (double)BYTE_LENGTH));
    // printf("newSize %d ", *bitSize);
    uint8_t* newArray = (uint8_t*)malloc(sizeof(uint8_t) * *bitSize);
    for (int j = 0; j < *bitSize; j++)
    { // cada linha da matriz
        for (int i = 0; i < BYTE_LENGTH; i++)
        {
            setBit(&newArray[j], i, (array[j * 8 + i] >> 0) & 1);
        }
    }

    return newArray;
}

int* decompressBitArray(uint8_t* array, int intSize)
{
    int bitSize = (int)ceil(((double)intSize / (double)BYTE_LENGTH));
    int* newArray = (int*)malloc(sizeof(int) * intSize);
    for (int j = 0; j < bitSize; j++)
    { // cada linha da matriz
        for (int i = 0; i < BYTE_LENGTH; i++)
        {
            setBitInteger(&newArray[j * 8 + i], 0, (array[j] >> i) & 1);
        }
    }

    return newArray;
}

void WriteFileCByte(FILE* file, cbyte a, lwe_instance lwe)
{
    int bitSize;
    for (int i = 0; i < BYTE_LENGTH; i++)
    { // cada bit do byte
        for (int j = 0; j < lwe.N; j++)
        { // cada linha da matriz
            uint8_t* m = compressBitArray(a[i][j], lwe.N, &bitSize);
            fwrite(m, sizeof(uint8_t), bitSize, file); // copia cada inteiro
        }
    }
}


cbyte ReadFileCByte(FILE* file, lwe_instance lwe)
{
    cbyte cb = (int***)malloc(sizeof(int**) * BYTE_LENGTH);

    for (int i = 0; i < BYTE_LENGTH; i++)
    {
        cb[i] = (int**)malloc(sizeof(int*) * lwe.N);

        for (int j = 0; j < lwe.N; j++)
        {
            int bitSize = (int)ceil(((double)lwe.N / (double)BYTE_LENGTH));
            uint8_t* m = (uint8_t*)malloc(sizeof(uint8_t) * bitSize);

            fread(m, sizeof(uint8_t), bitSize, file);

            cb[i][j] = decompressBitArray(m, lwe.N);
        }
    }

    return cb;
}