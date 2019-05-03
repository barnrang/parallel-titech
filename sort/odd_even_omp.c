#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// void swap_val(float &a, float &b) {
//     float tmp = a;
//     a = b;
//     b = tmp;
// }

long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int init(double *data, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    data[i] = (double)rand() / RAND_MAX;
  }
  return 0;
}

int check(double *data, int n)
{
  int i;
  int flag = 0;
  for (i = 0; i < n-1; i++) {
    if (data[i] > data[i+1]) {
      printf("Error: data[%d]=%.4lf, data[%d]=%.4lf\n",
	     i, data[i], i+1, data[i+1]);
      flag++;
    }
  }
  if (flag == 0) {
    printf("Data are sorted\n");
  }
  return 0;
}

void odd_even_sort(double *data, int n, int num_threads)
{
    float tmp = 0;
    int ex0,ex1=1;
    int i, j;
    int count = 0;

    for(i=0;i<(n/2+1);i++)
    {
            #pragma omp parallel for private(tmp,j) shared(data) num_threads(num_threads)
            for(j=0;j<n-1;j+=2)
            {
                if(data[j]> data[j+1])
                {
                    tmp = data[j];
                    data[j] = data[j+1];
                    data[j+1] = tmp;
                }
            }
        
        {
            #pragma omp parallel for private(tmp,j) shared(data) num_threads(num_threads)
            for(j = 1; j < n-1; j += 2)
            {
                if(data[j]> data[j+1])
                {
                    tmp = data[j];
                    data[j] = data[j+1];
                    data[j+1] = tmp;
                }
            }
        }
    }

    // while (ex1){
    //     ex0 = 0;
    //     ex1 = 0;
    //     #pragma omp parallel num_threads(num_threads) shared(data) private(j, tmp)
    //     {
    //         #pragma omp for 
    //         for(j = 0; j < n-1; j = j+2){
    //             //printf("%d 0\n",count);
    //             if(data[j] > data[j+1]) {
    //                 tmp = data[j];
    //                 data[j] = data[j+1];
    //                 data[j+1] = tmp;
    //                 ex0 = 1;
    //             }
    //         }
    //         if (ex0){
    //             #pragma omp for 
    //             for(j = 1; j < n-1; j = j+2){
    //                 //printf("%d 1\n",count);
    //                 if(data[j] > data[j+1]) {
    //                     tmp = data[j];
    //                     data[j] = data[j+1];
    //                     data[j+1] = tmp;
    //                     ex1 = 1;
    //                 }
    //             }
    //         }
    //     }
    //     count++;
    // }
}

int main(int argc, char *argv[])
{
  int n = 10000;
  double *data;
  int i;
  int num_threads = 4;

  if (argc >= 2) {
    n = atol(argv[1]);
  }
  if (argc >= 3) {
    num_threads = atol(argv[2]);
  }

  data = malloc(sizeof(double)*n);

  for (i = 0; i < 3; i++) {
    struct timeval st;
    struct timeval et;
    long us;

    init(data, n);
    /*print(data, n);*/
    gettimeofday(&st, NULL); /* get start time */
    odd_even_sort(data, n, num_threads);
    gettimeofday(&et, NULL); /* get start time */
    us = time_diff_us(st, et);

    printf("sorting %d data took %ld us\n",
	   n, us);

    check(data, n);
    /*print(data, n);*/
  }

  free(data);

  return 0;
}
