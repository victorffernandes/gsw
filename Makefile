
run-tests:
	gcc main_test.c -o builds/main_test.o -lm && ./builds/main_test.o

run-local:
	gcc main.c -o builds/main.o -lm && ./builds/main.o

build-process-bmp:
	gcc-13 image_process.c -o builds/process_bmp.o -lm

build-cipher-bmp:
	gcc-13 cipher_image.c -o builds/cipher_bmp.o  -lm

build-read-cipher-bmp:
	gcc-13 decipher_image.c -o builds/read_cipher_bmp.o -lm

build-all:
	make build-process-bmp
	make build-cipher-bmp
	make build-read-cipher-bmp

run-gpu:
	nvcc -g main.cu -o builds/main.o -lm && ./builds/main.o