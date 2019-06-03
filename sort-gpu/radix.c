#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/*
Radix sort
Sort Integer ranging from 0 - 255
*/

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

/* From stackoverflow */
int rand_lim(int limit) {
/* return a random number between 0 and limit inclusive.
 */

    int divisor = RAND_MAX/(limit+1);
    int retval;

    do { 
        retval = rand() / divisor;
    } while (retval > limit);

    return retval;
}

void make_data(int* arr, int N) {
    printf("Making data\n");
    for (int i = 0; i < N; i++){
        arr[i] = (int)((double)rand() / RAND_MAX * 4095);
    }
    printf("Finish making data\n");
}

void check_sort(int* arr, int N){
    for (int i = 0; i < N-1; i++) {
        if (arr[i] > arr[i+1]) printf("arr[%d] > arr[%d] - %d > %d\n", i, i+1, arr[i], arr[i+1]);
    }
}

int main(int argc, char *argv[]) {

    struct timeval st;
    struct timeval et;

    int N = 1000000;
    srand ( time(0) );

    if (argc >= 2) {
        N = atol(argv[1]);
    }
    int* arr = malloc(sizeof(int) * N);
    make_data(arr, N);

    // Prepare bucket
    printf("Prepared data\n");

    gettimeofday(&st, NULL); /* get start time */

    /* Search for Maximum */
    int* zero_bucket = malloc(sizeof(int) * N);
    int* one_bucket = malloc(sizeof(int) * N);
    int count_zero = 0, count_one = 0;
    int MAX = 0;
    for (int i = 0; i < N; i++){
        if (arr[i] > MAX) MAX = arr[i];
    }
    int step = (int)log2(MAX) + 1;
    printf("Max found\n");

    /* Loop through each bit digit */
    int LSB = 1;
    for (int i = 0; i < step; i++) {
        count_zero = 0; count_one = 0;
        for (int j = 0; j < N; j++) {
            if ((arr[j] & LSB) == 0) {
                zero_bucket[count_zero] = arr[j];
                count_zero++;
            } else {
                one_bucket[count_one] = arr[j];
                count_one++;
            }
        }
        for (int j = 0; j < count_zero; j++) {
            arr[j] = zero_bucket[j];
        }
        for (int j = 0; j < count_one; j++) {
            arr[j + count_zero] = one_bucket[j];
        }
        LSB *= 2;
    }

    gettimeofday(&et, NULL); /* get start time */

    long us = time_diff_us(st, et);

    printf("sorting %d data took %ld us\n",
    N, us);

    // check_sort(arr, N);
    return 0;
}