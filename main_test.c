#include <stdio.h>
#include <stdlib.h>
#include "gsw.c"


int TestKeyGen(){
    srand(4);
    int L = log2(q) + 1;
    int N = K * L;
    int m = 8;

    int * t = GenerateVector(K);
    int * secretKey = SecretKeyGen(t);
    int * v = Powersof2(secretKey, L);
    
    int ** publicKey = PublicKeyGen(t, m); // pubK [m, K+1]

    int * check = MultiplyVectorxMatrix(secretKey, publicKey, m, K+1); // check must be equal to error as A.s = e
    int check_assert[9] = {1,126,2,2,2,2,1,2,0};

    for(int h = 0; h < K+1; h++){
        if (mod(check[h], q) != check_assert[h]) return 0;
    }

    return 1;
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
        printf("passed %s test", test_name);
    } else{
        printf("failed %s test", test_name);
    }
}

int main(){
    AssertTest(TestKeyGen(), "TestKeyGen");
    // AssertTest(TestApplyRows(), "TestApplyRows");
} 

