# Use a imagem base do CUDA
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Instale dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Crie um diretório para a aplicação
WORKDIR /app

# Copie o código fonte para o container
COPY . /app

# Compile o código fonte CUDA (exemplo com um arquivo main.cu)
RUN make build-all

# Comando padrão para rodar a aplicação
CMD ["make", "run-tests"]
