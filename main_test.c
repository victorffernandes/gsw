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

int assertEqualsMatrix(int ** v1, int ** v2, int rows, int columns){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            if (v1[i][j] != v2[i][j]){
                return 0;
            }
        }
    }
    return 1;
}

int TestKeyGen(){
    srand(4);
    lwe_instance lwe_test = GenerateLweInstance(4);

    int * t = GenerateVector(lwe_test.n, lwe_test);
    int * secretKey = SecretKeyGen(t, lwe_test);
    int * v = Powersof2(secretKey, lwe_test);
    
    int ** publicKey = PublicKeyGen(t, lwe_test); // pubK [m, n+1]

    int * check = MultiplyVectorxMatrixOverQ(secretKey, publicKey, lwe_test.m, lwe_test.n+1, lwe_test.q); // check must be equal to error as A.s = e
    printVector(check, lwe_test.m, "check");
    int check_assert[20] = {1,1,1,1,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1};

    for(int h = 0; h < lwe_test.m; h++){
        printf("check: %d ", check[h]);
        if (mod(check[h], lwe_test.q) != check_assert[h]) return 0;
    }

    return 1;
}

// int TestInternalProductPowerof2AndBitDecomp(){
//     srand(4);
//     int L = log2(q) + 1;
//     int N = n * L;
//     int m = 8;

//     int * a = GenerateVector(L);
//     int * b = GenerateVector(L);
//     int * BitDecompA = BitDecomp(a, L);
//     int * Powersof2B = Powersof2(b, n+1, L);

//     int internalProductAxB = InternalProduct(a,b, L);
//     int internalProductBitDecompAxPowersof2B = InternalProduct(BitDecompA, Powersof2B, L * n );
//     return internalProductAxB == internalProductBitDecompAxPowersof2B;
// }

// int TestBitDecomp(){
//     srand(4);
//     int L = log2(q) + 1;
//     int N = (n+1) * L;
//     int m = 8;

//     int * a = GenerateVector(n);
//     int * BitDecompA = BitDecomp(a, L);
//     int * BitDecompInverseA = BitDecompInverse(BitDecompA, L*(n+1) );

//     return assertEqualsVector(a, BitDecompInverseA, n+1);
// }

// int TestInternalProduct(){
//     srand(4);
//     int L = log2(q) + 1;
//     int N = (n+1) * L;
//     int m = 8;

//     int * a_ = GenerateVector(N);
//     int * a = GenerateVector(n);
//     int * b = GenerateVector(n);
//     int * Powersof2B = Powersof2(b, n+1, L); // N-dimension
//     int * BitDecompInverseA_ = BitDecompInverse(a_, N); // // L-dimension
//     int * FlattenA_ = Flatten(a_, N); // N-dimension

//     int internalProductA_xPowerof2B = mod(InternalProduct(a_,Powersof2B, N ), q);
//     int internalProductBitDecompInverseA_xB = mod(InternalProduct(BitDecompInverseA_, b, n ),q);
//     int internalProductFlattenA_xPowersof2B = mod(InternalProduct(FlattenA_, Powersof2B,  N ), q);
    
//     int * BitDecompA = BitDecomp(a, L); // // N-dimension
//     int internalProductBitDecompAxPowersof2B = mod(InternalProduct(BitDecompA, Powersof2B,  N ), q);
//     int internalProductAxB = mod(InternalProduct(BitDecompA, Powersof2B,  N ), q);

//     return internalProductA_xPowerof2B == internalProductBitDecompInverseA_xB && 
//         internalProductBitDecompInverseA_xB == internalProductFlattenA_xPowersof2B && 
//         internalProductFlattenA_xPowersof2B == internalProductA_xPowerof2B &&
//         internalProductBitDecompAxPowersof2B == internalProductAxB;
// }


// int TestApplyRows(){
//     srand(4);
//     int L = log2(q) + 1;
//     int N = (n+1) * L;
//     int m = 8;
    
//     int ** sample = GenerateMatrix(4, n);
//     int ** result = applyRows(sample, 4, L, BitDecomp);
//     int ** result_ = applyRows(result, 4, N, BitDecompInverse);

//     return assertEqualsMatrix(sample, result_, 4 , 4);
// }

int TestEncrypt(){

    srand(4);
    lwe_instance lwe = GenerateLweInstance(25);

    int * t = GenerateVector(lwe.n, lwe);
    int * secretKey = SecretKeyGen(t, lwe);
    int * v = Powersof2(secretKey, lwe);
    
    int ** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    printVector(secretKey, lwe.n + 1, "secretkey");
    printMatrix(publicKey, lwe.m, lwe.n+1, "pubKey");
  
    int ** C1 = Encrypt(30, publicKey, lwe);
    int ** C2 = Encrypt(15, publicKey, lwe);
    int ** C3 = Encrypt(1, publicKey, lwe);
    int ** C4 = Encrypt(1, publicKey, lwe);

    int ** hSum = HomomorphicSum(C1, C2, lwe);
    int ** hMult = HomomorphicMult(C1, C2, lwe);
    int ** hMultConst = HomomorphicMultByConst(C1, 2, lwe);
    int ** hNAND = HomomorphicNAND(C3, C4, lwe); 

    int sumValue = MPDecrypt(hSum, v, lwe);
    int multValue = MPDecrypt(hMult, v, lwe);
    int multConstValue = MPDecrypt(hMultConst, v, lwe);
    int NANDValue = Decrypt(hNAND, v, lwe);

    printf("\n sumValue[%d] multValue[%d] multConstValue[%d] NANDValue[%d] ", sumValue, multValue, multConstValue, NANDValue);

    
    // Flatten (m*In + BitDecomp(R * A))

    //TODO: free

    return 3;
}

void AssertTest(int result, char * test_name){
    if(result){
        printf("passed %s \n", test_name);
    } else{
        printf("failed %s \n", test_name);
    }
}

int main(){
    // AssertTest(TestBitDecomp(), "TestBitDecomp");
    // AssertTest(TestKeyGen(), "TestKeyGen");
    // AssertTest(TestInternalProductPowerof2AndBitDecomp(), "TestInternalProductPowerof2AndBitDecomp");
    // AssertTest(TestInternalProduct(), "TestInternalProduct");
    // AssertTest(TestApplyRows(), "TestApplyRows");
    AssertTest(TestEncrypt(), "TestEncrypt");
} 

