
#include <stdio.h>
#include <stdlib.h>
#include "lib/gsw.c"

int assertEqualsVector(int* v1, int* v2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (v1[i] != v2[i])
        {
            return 0;
        }
    }
    return 1;
}

int assertEqualsMatrix(int** v1, int** v2, int rows, int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            if (v1[i][j] != v2[i][j])
            {
                return 0;
            }
        }
    }
    return 1;
}

int TestKeyGen(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe_test = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe_test.n, lwe_test); // [n]
    int* secretKey = SecretKeyGen(t, lwe_test); // [n + 1]

    int** publicKey = PublicKeyGen(t, lwe_test); // pubK [m, n+1]

    int* check = MultiplyVectorxMatrixOverQ(secretKey, publicKey, lwe_test.m, lwe_test.n + 1, lwe_test.q); // check must be equal to error as A.s = e
    printVector(check, lwe_test.m, "CHECK");

    return 1;
}

int TestMod(BitSecurity bitSecurity)
{
    srand(4);
    int q = 512;

    int r1 = mod(513, q);
    int r2 = mod(1024, q);
    int r3 = mod(-511, q);
    int r4 = mod(-513, q);
    int r5 = mod(-1024, q);

    printf("r1: %d, r2: %d, r3: %d, r4: %d, r5: %d \n", r1, r2, r3, r4, r5);

    return 1;
}

int TestInternalProductPowerof2AndBitDecomp(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* a = GenerateVector(lwe.n, lwe);
    int* b = GenerateVector(lwe.n, lwe);
    int* BitDecompA = BitDecomp(a, lwe.n, lwe);
    int* Powersof2B = Powersof2(b, lwe); // [(lwe.n+1) * lwe.l]

    int internalProductAxB = InternalProduct(a, b, lwe.n);
    int internalProductBitDecompAxPowersof2B = InternalProduct(BitDecompA, Powersof2B, lwe.N);
    return internalProductAxB == internalProductBitDecompAxPowersof2B;
}

int TestBitDecomp(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* a = GenerateVector(lwe.n + 1, lwe);
    int* BitDecompA = BitDecomp(a, lwe.n + 1, lwe);
    int* BitDecompInverseA = BitDecompInverse(BitDecompA, lwe.N, lwe);

    return assertEqualsVector(a, BitDecompInverseA, lwe.n + 1);
}

int TestInternalProduct(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* a_ = GenerateVector(lwe.N, lwe);
    int* a = GenerateVector(lwe.n + 1, lwe);
    int* b = GenerateVector(lwe.n, lwe);
    int* Powersof2B = Powersof2(b, lwe);                        // N-dimension
    int* BitDecompInverseA_ = BitDecompInverse(a_, lwe.N, lwe); // // L-dimension
    int* FlattenA_ = Flatten(a_, lwe.N, lwe);                   // N-dimension

    int internalProductA_xPowerof2B = mod(InternalProduct(a_, Powersof2B, lwe.N), lwe.q);
    int internalProductBitDecompInverseA_xB = mod(InternalProduct(BitDecompInverseA_, b, lwe.n + 1), lwe.q);
    int internalProductFlattenA_xPowersof2B = mod(InternalProduct(FlattenA_, Powersof2B, lwe.N), lwe.q);

    int* BitDecompA = BitDecomp(a, lwe.n + 1, lwe); // N-dimension
    int internalProductBitDecompAxPowersof2B = mod(InternalProduct(BitDecompA, Powersof2B, lwe.N), lwe.q);
    int internalProductAxB = mod(InternalProduct(BitDecompA, Powersof2B, lwe.N), lwe.q);

    return internalProductA_xPowerof2B == internalProductBitDecompInverseA_xB &&
        internalProductBitDecompInverseA_xB == internalProductFlattenA_xPowersof2B &&
        internalProductFlattenA_xPowersof2B == internalProductA_xPowerof2B &&
        internalProductBitDecompAxPowersof2B == internalProductAxB;
}

int TestApplyRows(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int** sample = GenerateMatrixOverQ(4, 4, lwe.q);
    int** result = applyRows(sample, 4, 4, &BitDecomp, lwe);
    int** result_ = applyRows(result, 4, 4 * lwe.l, &BitDecompInverse, lwe);

    return assertEqualsMatrix(sample, result_, 4, 4);
}

int TestNOT(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int** C4 = Encrypt(0, publicKey, lwe);
    int** C5 = Encrypt(1, publicKey, lwe);

    int NOT1Value = Decrypt(HomomorphicNOT(C4, lwe), v, lwe);
    int NOT2Value = Decrypt(HomomorphicNOT(C5, lwe), v, lwe);

    return NOT1Value == 1 && NOT2Value == 0;
}

int TestNAND(BitSecurity bitSecurity)
{
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    printf("t: %p, secretKey: %p, v: %p\n", t, secretKey, v);


    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int** C4 = Encrypt(1, publicKey, lwe);
    int** C5 = Encrypt(0, publicKey, lwe);

    printf("publicKey: %p, C4: %p, C5: %p\n", publicKey, C4, C5);


    int** Identity = GenerateIdentity(lwe.N, lwe.N);

    int NAND1Value = Decrypt(HomomorphicNAND(C5, C5, Identity, lwe), v, lwe); // 0 0 1
    int NAND2Value = Decrypt(HomomorphicNAND(C4, C5, Identity, lwe), v, lwe); // 1 0 1
    int NAND3Value = Decrypt(HomomorphicNAND(C5, C4, Identity, lwe), v, lwe); // 0 1 1 ok
    int NAND4Value = Decrypt(HomomorphicNAND(C4, C4, Identity, lwe), v, lwe); // 1 1 0 ok

    printf("NAND1Value: %d, NAND2Value: %d, NAND3Value: %d, NAND4Value: %d\n", NAND1Value, NAND2Value, NAND3Value, NAND4Value);


    return NAND1Value == 1 && NAND2Value == 1 && NAND3Value == 1 && NAND4Value == 0;
}

int TestAND(BitSecurity bitSecurity)
{
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    printf("t: %p, secretKey: %p, v: %p\n", t, secretKey, v);


    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int** C4 = Encrypt(1, publicKey, lwe);
    int** C5 = Encrypt(0, publicKey, lwe);

    printf("publicKey: %p, C4: %p, C5: %p\n", publicKey, C4, C5);


    int AND1Value = Decrypt(HomomorphicAND(C5, C5, lwe), v, lwe); // 0 0 0
    int AND2Value = Decrypt(HomomorphicAND(C4, C5, lwe), v, lwe); // 1 0 0
    int AND3Value = Decrypt(HomomorphicAND(C5, C4, lwe), v, lwe); // 0 1 0 ok
    int AND4Value = Decrypt(HomomorphicAND(C4, C4, lwe), v, lwe); // 1 1 1 ok

    printf("AND1Value: %d, AND2Value: %d, AND3Value: %d, AND4Value: %d\n", AND1Value, AND2Value, AND3Value, AND4Value);


    return AND1Value == 0 && AND2Value == 0 && AND3Value == 0 && AND4Value == 1;
}

int TestXOR(BitSecurity bitSecurity)
{
    // srand(5);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int** C4 = Encrypt(1, publicKey, lwe);
    int** C5 = Encrypt(0, publicKey, lwe);

    int XOR1Value = Decrypt(HomomorphicXOR(C5, C5, lwe), v, lwe);
    int XOR2Value = Decrypt(HomomorphicXOR(C4, C5, lwe), v, lwe);
    int XOR3Value = Decrypt(HomomorphicXOR(C5, C4, lwe), v, lwe);
    int XOR4Value = Decrypt(HomomorphicXOR(C4, C4, lwe), v, lwe);

    return XOR3Value == 1 && XOR1Value == 0 && XOR2Value == 1 && XOR4Value == 0;
}

int TestEncrypt(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    int** C1 = Encrypt(0, publicKey, lwe);
    int** C2 = Encrypt(1, publicKey, lwe);
    int** C3 = Encrypt(1, publicKey, lwe);
    int** C4 = Encrypt(1, publicKey, lwe);
    int** C5 = Encrypt(0, publicKey, lwe);

    int  DC1 = Decrypt(C1, v, lwe);
    int  DC2 = Decrypt(C2, v, lwe);
    int  DC3 = Decrypt(C3, v, lwe);
    int  DC4 = Decrypt(C4, v, lwe);
    int  DC5 = Decrypt(C5, v, lwe);

    printf("\n %d %d %d %d %d \n", DC1, DC2, DC3, DC4, DC5);

    return DC1 == 0 && DC2 == 1 && DC3 == 1 && DC4 == 1 && DC5 == 0;
}

int TestWriteCByte(BitSecurity bitSecurity)
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(bitSecurity);

    int* t = GenerateVector(lwe.n, lwe);
    int* secretKey = SecretKeyGen(t, lwe);
    int* v = Powersof2(secretKey, lwe);

    int** publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    FILE* outputFile = fopen("sample", "wb");

    cbyte cb = ByteEncrypt(30, publicKey, lwe);
    WriteFileCByte(outputFile, cb, lwe);
    fclose(outputFile);

    FILE* inputfile = fopen("sample", "rb");
    cbyte cb2 = ReadFileCByte(inputfile, lwe);
    // printMatrix(cb2[0], lwe.N, lwe.N, "matrix read file: \n");
    fclose(inputfile);

    byte a = ByteDecrypt(cb2, v, lwe);

    return a == 30;
}

void AssertTest(int result, char* test_name)
{
    if (result)
    {
        printf("\n passed %s \n", test_name);
    }
    else
    {
        printf("\n failed %s \n", test_name);
    }
}

int main()
{
    BitSecurity bit = SECURITY_128;
    AssertTest(TestBitDecomp(bit), "TestBitDecomp");
    AssertTest(TestKeyGen(bit), "TestKeyGen");
    AssertTest(TestMod(bit), "TestMod");
    AssertTest(TestInternalProductPowerof2AndBitDecomp(bit), "TestInternalProductPowerof2AndBitDecomp");
    AssertTest(TestInternalProduct(bit), "TestInternalProduct");
    AssertTest(TestApplyRows(bit), "TestApplyRows");
    AssertTest(TestEncrypt(bit), "TestEncrypt");
    AssertTest(TestAND(bit), "TestAND");
    AssertTest(TestNAND(bit), "TestNAND");

    for (int i = 0; i < 5; i++)
    {
        AssertTest(TestXOR(bit), "TestXOR");
    }
    AssertTest(TestNOT(bit), "TestNOT");
    AssertTest(TestWriteCByte(bit), "TestWriteCByte");
}
