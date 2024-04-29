
#include <stdio.h>
#include <stdlib.h>
#include "lib/gsw.c"

int assertEqualsVector(int *v1, int *v2, int size)
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

int assertEqualsMatrix(int **v1, int **v2, int rows, int columns)
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

int TestKeyGen()
{
    srand(4);
    lwe_instance lwe_test = GenerateLweInstance(2);

    int *t = GenerateVector(lwe_test.n, lwe_test);
    int *secretKey = SecretKeyGen(t, lwe_test);
    int *v = Powersof2(secretKey, lwe_test);

    int **publicKey = PublicKeyGen(t, lwe_test); // pubK [m, n+1]

    int *check = MultiplyVectorxMatrixOverQ(secretKey, publicKey, lwe_test.m, lwe_test.n + 1, lwe_test.q); // check must be equal to error as A.s = e
    printVector(check, lwe_test.m, "CHECK");
    int check_assert[32] = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};

    for (int h = 0; h < lwe_test.m; h++)
    {
        if (mod(check[h], lwe_test.q) != check_assert[h])
            return 0;
    }

    return 1;
}

int TestInternalProductPowerof2AndBitDecomp()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(5);

    int *a = GenerateVector(lwe.n, lwe);
    int *b = GenerateVector(lwe.n, lwe);
    int *BitDecompA = BitDecomp(a, lwe.n, lwe);
    int *Powersof2B = Powersof2(b, lwe); // [(lwe.n+1) * lwe.l]

    int internalProductAxB = InternalProduct(a, b, lwe.n);
    int internalProductBitDecompAxPowersof2B = InternalProduct(BitDecompA, Powersof2B, lwe.N);
    return internalProductAxB == internalProductBitDecompAxPowersof2B;
}

int TestBitDecomp()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(5);

    int *a = GenerateVector(lwe.n + 1, lwe);
    int *BitDecompA = BitDecomp(a, lwe.n + 1, lwe);
    int *BitDecompInverseA = BitDecompInverse(BitDecompA, lwe.N, lwe);

    return assertEqualsVector(a, BitDecompInverseA, lwe.n + 1);
}

int TestInternalProduct()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(5);

    int *a_ = GenerateVector(lwe.N, lwe);
    int *a = GenerateVector(lwe.n + 1, lwe);
    int *b = GenerateVector(lwe.n, lwe);
    int *Powersof2B = Powersof2(b, lwe);                        // N-dimension
    int *BitDecompInverseA_ = BitDecompInverse(a_, lwe.N, lwe); // // L-dimension
    int *FlattenA_ = Flatten(a_, lwe.N, lwe);                   // N-dimension

    int internalProductA_xPowerof2B = mod(InternalProduct(a_, Powersof2B, lwe.N), lwe.q);
    int internalProductBitDecompInverseA_xB = mod(InternalProduct(BitDecompInverseA_, b, lwe.n + 1), lwe.q);
    int internalProductFlattenA_xPowersof2B = mod(InternalProduct(FlattenA_, Powersof2B, lwe.N), lwe.q);

    int *BitDecompA = BitDecomp(a, lwe.n + 1, lwe); // N-dimension
    int internalProductBitDecompAxPowersof2B = mod(InternalProduct(BitDecompA, Powersof2B, lwe.N), lwe.q);
    int internalProductAxB = mod(InternalProduct(BitDecompA, Powersof2B, lwe.N), lwe.q);

    return internalProductA_xPowerof2B == internalProductBitDecompInverseA_xB &&
           internalProductBitDecompInverseA_xB == internalProductFlattenA_xPowersof2B &&
           internalProductFlattenA_xPowersof2B == internalProductA_xPowerof2B &&
           internalProductBitDecompAxPowersof2B == internalProductAxB;
}

int TestApplyRows()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(5);

    int **sample = GenerateMatrixOverQ(4, 4, lwe.q);
    int **result = applyRows(sample, 4, 4, &BitDecomp, lwe);
    int **result_ = applyRows(result, 4, 4 * lwe.l, &BitDecompInverse, lwe);

    return assertEqualsMatrix(sample, result_, 4, 4);
}

int TestNOT()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(25);

    int *t = GenerateVector(lwe.n, lwe);
    int *secretKey = SecretKeyGen(t, lwe);
    int *v = Powersof2(secretKey, lwe);

    int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int **C4 = Encrypt(0, publicKey, lwe);
    int **C5 = Encrypt(1, publicKey, lwe);

    int NOT1Value = Decrypt(HomomorphicNOT(C4, lwe), v, lwe);
    int NOT2Value = Decrypt(HomomorphicNOT(C5, lwe), v, lwe);

    return NOT1Value == 1 && NOT2Value == 0;
}

int TestNAND()
{
    lwe_instance lwe = GenerateLweInstance(25);

    int *t = GenerateVector(lwe.n, lwe);
    int *secretKey = SecretKeyGen(t, lwe);
    int *v = Powersof2(secretKey, lwe);

    int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int **C4 = Encrypt(1, publicKey, lwe);
    int **C5 = Encrypt(0, publicKey, lwe);

    int ** Identity = GenerateIdentityMatrix(lwe.N, lwe.N);

    int NAND1Value = Decrypt(HomomorphicNAND(C5, C5, Identity, lwe), v, lwe);
    int NAND2Value = Decrypt(HomomorphicNAND(C4, C5, Identity, lwe), v, lwe);
    int NAND3Value = Decrypt(HomomorphicNAND(C5, C4, Identity, lwe), v, lwe);
    int NAND4Value = Decrypt(HomomorphicNAND(C4, C4, Identity, lwe), v, lwe);

    return NAND1Value == 1 && NAND2Value == 1 && NAND3Value == 1 && NAND4Value == 0;
}

int TestXOR()
{
    // srand(5);
    lwe_instance lwe = GenerateLweInstance(31);

    int *t = GenerateVector(lwe.n, lwe);
    int *secretKey = SecretKeyGen(t, lwe);
    int *v = Powersof2(secretKey, lwe);

    int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]
    int **C4 = Encrypt(1, publicKey, lwe);
    int **C5 = Encrypt(0, publicKey, lwe);

    int XOR1Value = Decrypt(HomomorphicXOR(C5, C5, lwe), v, lwe);
    int XOR2Value = Decrypt(HomomorphicXOR(C4, C5, lwe), v, lwe);
    int XOR3Value = Decrypt(HomomorphicXOR(C5, C4, lwe), v, lwe);
    int XOR4Value = Decrypt(HomomorphicXOR(C4, C4, lwe), v, lwe);

    return XOR3Value == 1 && XOR1Value == 0 && XOR2Value == 1 && XOR4Value == 0;
}

int TestEncrypt()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(30);

    int *t = GenerateVector(lwe.n, lwe);
    int *secretKey = SecretKeyGen(t, lwe);
    int *v = Powersof2(secretKey, lwe);

    int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    int **C1 = Encrypt(30, publicKey, lwe);
    int **C2 = Encrypt(15, publicKey, lwe);
    int **C3 = Encrypt(1, publicKey, lwe);
    int **C4 = Encrypt(1, publicKey, lwe);
    int **C5 = Encrypt(0, publicKey, lwe);

    int **hSum = HomomorphicSum(C1, C2, lwe);
    int **hMult = HomomorphicMult(C1, C2, lwe);
    int **hMultConst = HomomorphicMultByConst(C1, 2, lwe);
    int **hNAND = HomomorphicNAND(C3, C4, lwe);
    int **hXOR1 = HomomorphicXOR(C3, C4,lwe);
    int **hXOR2 = HomomorphicXOR(C3, C5,lwe);

    int sumValue = MPDecrypt(hSum, v, lwe);
    int multValue = MPDecrypt(hMult, v, lwe);
    int multConstValue = MPDecrypt(hMultConst, v, lwe);
    int NANDValue = Decrypt(hNAND, v, lwe);

    int XOR1Value = Decrypt(hXOR1, v, lwe);
    int XOR2Value = Decrypt(hXOR2, v, lwe);

    printf("\n %d %d \n", XOR1Value, XOR2Value);

    return sumValue == 44 & multConstValue == 60 && multValue == 450 && XOR1Value == 0 && XOR2Value == 1;
}

int TestWriteCByte()
{
    srand(4);
    lwe_instance lwe = GenerateLweInstance(30);

    int *t = GenerateVector(lwe.n, lwe);
    int *secretKey = SecretKeyGen(t, lwe);
    int *v = Powersof2(secretKey, lwe);

    int **publicKey = PublicKeyGen(t, lwe); // pubK [m, n+1]

    FILE *outputFile = fopen("sample", "wb");

    cbyte cb = ByteEncrypt(30, publicKey, lwe);
    WriteFileCByte(outputFile, cb, lwe);
    fclose(outputFile);

    FILE *inputfile = fopen("sample", "rb");
    cbyte cb2 = ReadFileCByte(inputfile, lwe);
    // printMatrix(cb2[0], lwe.N, lwe.N, "matrix read file: \n");
    fclose(inputfile);

    byte a = ByteDecrypt(cb2, v, lwe);

    return a == 30;
}

void AssertTest(int result, char *test_name)
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
    AssertTest(TestBitDecomp(), "TestBitDecomp");
    AssertTest(TestKeyGen(), "TestKeyGen");
    AssertTest(TestInternalProductPowerof2AndBitDecomp(), "TestInternalProductPowerof2AndBitDecomp");
    AssertTest(TestInternalProduct(), "TestInternalProduct");
    AssertTest(TestApplyRows(), "TestApplyRows");
    // AssertTest(TestEncrypt(), "TestEncrypt");
    AssertTest(TestNAND(), "TestNAND");

    for (int i = 0; i < 5; i++)
    {
        AssertTest(TestXOR(), "TestXOR");
    }
    AssertTest(TestNOT(), "TestNOT");
    AssertTest(TestWriteCByte(), "TestWriteCByte");
}
