
run-tests:
	gcc -g main_test.c -o main_test.o -lm && ./main_test.o

run-local:
	gcc -g main.c -o main.o -lm && ./main.o