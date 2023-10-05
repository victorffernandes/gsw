#include <stdio.h>
#include <stdlib.h>
#include "gsw.c"

int assertEqualsVector(int * v1, int * v2, int size){
    for (int i = 0; i < size; i++){
        if (v1[i] != v2[i]){
            return 0;
        }
    }
    return 1;
}

int TestKeyGen(){
    srand(4);
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);
    
    int ** publicKey = PublicKeyGen(t, m); // pubK [m, K+1]

    int * check = MultiplyVectorxMatrixOverQ(secretKey, publicKey, m, K+1); // check must be equal to error as A.s = e
    int check_assert[9] = {1,126,2,2,2,2,1,2,0};

    for(int h = 0; h < K+1; h++){
        if (mod(check[h], q) != check_assert[h]) return 0;
    }

    return 1;
}

int TestInternalProductPowerof2AndBitDecomp(){
    srand(4);
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * a = GenerateVector(L);
    int * b = GenerateVector(L);
    int * BitDecompA = BitDecomp(a, L);
    int * Powersof2B = Powersof2(b, L);

    int internalProductAxB = InternalProduct(a,b, L);
    int internalProductBitDecompAxPowersof2B = InternalProduct(BitDecompA, Powersof2B, L * K );
    return internalProductAxB == internalProductBitDecompAxPowersof2B;
}

int TestBitDecomp(){
    srand(4);
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * a = GenerateVector(K);
    int * BitDecompA = BitDecomp(a, L);
    int * BitDecompInverseA = BitDecompInverse(BitDecompA, L*K);

    return assertEqualsVector(a, BitDecompInverseA, K);
}

int TestInternalProduct(){
    srand(4);
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * a = GenerateVector(N);
    int * b = GenerateVector(K);
    int * Powersof2B = Powersof2(b, L); // N-dimension
    int * BitDecompInverseA = BitDecompInverse(a, N); // // L-dimension
    int * FlattenA = Flatten(a, N); // N-dimension

    int internalProductAxPowerof2B = mod(InternalProduct(a,Powersof2B, N ), q);
    int internalProductBitDecompInverseAxB = mod(InternalProduct(BitDecompInverseA, b, K ),q);
    int internalProductFlattenAxPowersof2B = mod(InternalProduct(FlattenA, Powersof2B,  N ), q);

    return internalProductAxPowerof2B == internalProductBitDecompInverseAxB && 
        internalProductBitDecompInverseAxB == internalProductFlattenAxPowersof2B;
}

// int TestApplyRows(){
//     srand(4);
//     int L = log2(q) + 1;
//     int N = K * L;
//     int m = 8;

    
//     int sample[3] = {1,2,3};

//     // for(int h = 0; h < 2; h++){
//     //     for(int l = 0; l < 2; l++){
//     //         printf("%d ", sample[h][l]);
//     //     }
//     // }
//     int columns = 3 * K;
//     int ** result = applyRows(sample, 2,2, BitDecomp);



//     // for(int h = 0; h < 2; h++){
//     //     for(int l = 0; l < columns; l++){
//     //         printf("%d ", sample[h][l]);
//     //     }
//     // }


//     return 1;
// }

void AssertTest(int result, char * test_name){
    if(result){
        printf("passed %s \n", test_name);
    } else{
        printf("failed %s \n", test_name);
    }
}

int main(){
    AssertTest(TestBitDecomp(), "TestBitDecomp");
    AssertTest(TestKeyGen(), "TestKeyGen");
    AssertTest(TestInternalProductPowerof2AndBitDecomp(), "TestInternalProductPowerof2AndBitDecomp");
    AssertTest(TestInternalProduct(), "TestInternalProduct");
    // AssertTest(TestApplyRows(), "TestApplyRows");
} 

