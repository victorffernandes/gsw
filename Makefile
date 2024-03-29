
run-tests:
	gcc -g main_test.c -o builds/main_test.o -lm && ./builds/main_test.o

run-local:
	gcc -g main.c -o builds/main.o -lm && ./builds/main.o

run-bmp:
	gcc-13 -g image_process.c -o builds/bmp.o -lm && ./builds/bmp.o

run-gpu:
	nvcc -g main.cu -o builds/main.o -lm && ./builds/main.o