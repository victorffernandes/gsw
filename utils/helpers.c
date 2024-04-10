#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int formatKey(char str[], int *key, int n)
{
        char *token = strtok(str, " ");
        int i = 0;
        while (token != NULL)
        {
                key[i] = atoi(token);
                token = strtok(NULL, " ");
                i++;
        }

        return i == n;
}