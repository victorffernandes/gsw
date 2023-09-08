
run-tests:
	gcc -g main_test.c -o main_test -lm && ./main_test

run-local:
	gcc -g main.c -o main -lm && ./main