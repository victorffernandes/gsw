#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int mod(int m, int l) {
    if (m < 0) {
        return (l - (-m % (l))) % l;
    }

    return m % l;
}

int rand_ringz(int q) {
    return mod(rand(), q); // (rand() % (upper - lower + 1)) + (lower);
}

int rand_error(int max_error) {
    double u1, u2, z;

    // Generate two random numbers uniformly distributed in (0,1)
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);

    // Apply the Box-Muller transform
    z = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);

    // Scale the result and round to the nearest integer
    int result = round(z * max_error / 6.0); // 6.0 is an empirical factor for scaling

    // Bound the result to ensure it stays within [0, B]
    result = result < 0 ? 0 : (result > max_error ? max_error : result);

    return result;
}

void printMatrix(int** m, int r, int c, char* label) {
    printf("\n %s \n", label);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%d ", m[i][j]);
        }
        printf("\n ");
    }
    printf("\n");
}

void printVector(int* v, int size, char* label) {
    printf("\n %s ", label);
    for (int i = 0; i < size; i++) {
        printf("%d ", v[i]);
    }
    printf("\n");
}

void printVectorf(float* v, int size, char* label) {
    printf("\n %s ", label);
    for (int i = 0; i < size; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

int* GenerateErrorVector(int size, int max_error) {
    int* vector = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        vector[i] = rand_error(max_error);
    }

    return vector;
}

int InternalProduct(int* v1, int* v2, int size) {
    int value = 0;

    for (int i = 0; i < size; i++) {
        value += v1[i] * v2[i];
    }

    return value;
}

int* MultiplyVectorxVector(int* v1, int* v2, int size) {
    int* result = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        result[i] = v1[i] * v2[i];
    }

    return result;
}

float* DivideVectorxVector(int* v1, int* v2, float size) {
    float* result = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        result[i] = (float)v1[i] / (float)v2[i];
    }

    return result;
}

int** MultiplyMatrixEscalarOverQ(int e, int** matrix, int r, int c, int q) {
    int** result = (int**)malloc(sizeof(int*) * r);

    for (int i = 0; i < r; i++) {
        result[i] = (int*)malloc(sizeof(int) * c);
        for (int j = 0; j < c; j++) {
            result[i][j] = mod(matrix[i][j] * e, q);
        }
    }

    return result;
}

int* MultiplyVectorxMatrix(int* v, int** matrix, int r, int c) {
    int* result = (int*)malloc(sizeof(int) * r);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            result[i] += v[j] * matrix[i][j];
        }
        result[i] = result[i];
    }
    return result;
}

int* MultiplyVectorxMatrixOverQ(int* v, int** matrix, int r, int c, int q) {
    int* result = (int*)malloc(sizeof(int) * r);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            result[i] += v[j] * matrix[i][j];
        }
        result[i] = mod(result[i], q);
    }
    return result;
}

int** SumMatrixxMatrix(int** m1, int** m2, int r, int c) {
    int** result = (int**)malloc(sizeof(int*) * r);

    for (int i = 0; i < r; i++) {
        result[i] = (int*)malloc(sizeof(int) * c);
        for (int j = 0; j < c; j++) {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return result;
}

int** SubMatrixxMatrix(int** m1, int** m2, int r, int c) {
    int** result = (int**)malloc(sizeof(int*) * r);

    for (int i = 0; i < r; i++) {
        result[i] = (int*)malloc(sizeof(int) * c);
        for (int j = 0; j < c; j++) {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return result;
}

int** MultiplyMatrixxMatrixOverQ(int** m1, int** m2, int r1, int c1, int c2, int q) {
    int** result = (int**)malloc(sizeof(int*) * r1);

    for (int i = 0; i < r1; i++) {
        result[i] = (int*)malloc(sizeof(int) * c2);
        for (int j = 0; j < c2; j++) {
            int sum = 0;
            for (int z = 0; z < c1; z++) {
                sum += (m1[i][z]) * (m2[z][j]);
            }

            result[i][j] = mod(sum, q);
        }
    }

    return result;
}

int* SumVectorOverQ(int* v1, int* v2, int size, int q) {
    int* result = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        result[i] = (v1[i] + v2[i]) % q;
    }

    return result;
}



int** GenerateMatrixOverQ(int  rows, int columns, int q) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = rand_ringz(q);
        }
    }

    return matrix;
}

int** GenerateEmpty(int  rows, int columns) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
    }

    return matrix;
}

int** GenerateIdentityMultipliedByConst(int rows, int columns, int constant) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++) {
            if (i == j) {
                matrix[i][j] = constant;
            }
            else {
                matrix[i][j] = 0;
            }
        }
    }

    return matrix;
}

int** GenerateIdentityMultipliedByConstD(int rows, int columns, int constant) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++) {
            if (i == j) {
                matrix[i][j] = constant;
            }
            else {
                matrix[i][j] = 0;
            }
        }
    }

    return matrix;
}

int** GenerateIdentity(int rows, int columns) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++) {
            if (i == j) {
                matrix[i][j] = 1;
            }
            else {
                matrix[i][j] = 0;
            }
        }
    }

    return matrix;
}

int** GenerateBinaryMatrix(int rows, int columns) {
    int** matrix = (int**)malloc(sizeof(int*) * rows);

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(sizeof(int) * columns);
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = rand() % 2;// TODO: get int by parameter q
        }
    }

    return matrix;
}

int* GenerateBinaryVector(int size) {
    int* vector = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 2;// TODO: get int by parameter q
    }

    return vector;
}

void FreeMatrix(int** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
}