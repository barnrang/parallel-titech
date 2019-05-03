#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int main(int argc, char *argv[]){
    int num_threads = atol(argv[1]);
    int N = 1000000000;
    int* A;
    int i;
    A = malloc(sizeof(int) * N);
    for (i = 0; i < N; i++) A[i] = i;
    struct timeval st;
    struct timeval et;
    #pragma omp parallel for num_threads(num_threads)
    for (i = 0; i < N; i+=2) {
        A[i+1] += 500;
    }

    gettimeofday(&st, NULL); /* get start time */
    #pragma omp parallel for num_threads(num_threads)
    for (i = 0; i < N; i+=2) {
        A[i+1] += 500;
    }
    gettimeofday(&et, NULL); /* get start time */
    long us = time_diff_us(st, et);

    printf("time %ld\n", us);

}

