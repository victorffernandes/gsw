
run-tests:
	nvcc -G tests.c -o builds/main_test.o -lm && ./builds/main_test.o

run-local:
	nvcc -G main.c -o builds/main.o -lm && ./builds/main.o

build-process-bmp:
	nvcc image_process.cu -o builds/process_bmp.o -lm --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80

build-cipher-bmp:
	nvcc cipher_image.cu -o builds/cipher_bmp.o  -lm --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80

build-read-cipher-bmp:
	nvcc decipher_image.c -o builds/read_cipher_bmp.o -lm --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80

build-all:
	make build-process-bmp
	make build-cipher-bmp
	make build-read-cipher-bmp

run-gpu:
	nvcc lib/acc_gsw.cu -o builds/gpu.o -lm -G --forward-unknown-to-host-linker && ./builds/gpu.o