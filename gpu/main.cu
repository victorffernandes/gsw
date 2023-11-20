#include <stdio.h>

// Função do kernel para realizar a redução
__global__ void reduce(int ** matrix, int rows, int columns) {
    



}

int main() {
    // Tamanho do vetor
    int size = 2048;
    int h_input[size];

    for (int i = 0; i < size; i++){
        h_input[i] = i;
    }

    // Vetores no device (GPU)
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copia o vetor de entrada do host para o device
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (size + 512 - 1) / 512;

    // Chama o kernel com 1 bloco e size threads
    reduce<<<num_blocks, 512>>>(d_input, d_output, size);

    // Copia o resultado de volta do device para o host
    int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Imprime o resultado
    printf("Resultado: %d\n", h_output);

    // Libera a memória alocada no device
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
