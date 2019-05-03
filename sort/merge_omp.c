#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

void merge_sort(double* data, double* output, int s, int e, int thresh)
{
    // Do nothing
    if ((e - s) == 0) return;
    // Split the mid
    int mid = (e + s) / 2;

    // If too "small", serialize
    if ((e - s) > thresh)
    {
        #pragma omp task
        {merge_sort(data, output, mid+1, e, thresh);}
        #pragma omp task
        {merge_sort(data, output, s, mid, thresh);}
        #pragma omp taskwait
    } 
    else 
    {
        merge_sort(data, output, mid+1, e, thresh);
        merge_sort(data, output, s, mid, thresh);
    }

    // Merge part
    int p1 = s, p2 = mid+1;
    int p = s;
    while ((p1 <= mid) || (p2 <= e))
    {
        if (p1 > mid) {
            output[p] = data[p2];
            p2++;
        } else if (p2 > e) {
            output[p] = data[p1];
            p1++;
        } else if (data[p1] > data[p2]) {
            output[p] = data[p2];
            p2++;
        } else {
            output[p] = data[p1];
            p1++;
        }
        p++;
    }
    
    for (int i = s; i <= e; i++) data[i] = output[i];
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

int main(int argc, char *argv[])
{
  int n = 10000;
  double *data;
  double *output;
  int i, num_threads=4;
  int thresh = 2000;

  if (argc >= 2) {
    n = atol(argv[1]);
  }
  if (argc >= 3) {
    num_threads = atol(argv[2]);
  }
  if (argc >= 4) {
    thresh = atol(argv[3]);
  }

  data = malloc(sizeof(double)*n);
  output = malloc(sizeof(double)*n);

  for (i = 0; i < 3; i++) {
    struct timeval st;
    struct timeval et;
    long us;
    double res;

    init(data, n);
    /*print(data, n);*/
    gettimeofday(&st, NULL); /* get start time */
    #pragma omp parallel num_threads(num_threads)
    {
      #pragma omp single
      {
        merge_sort(data, output, 0, n, thresh);
      }
    }
    gettimeofday(&et, NULL); /* get start time */
    us = time_diff_us(st, et);

    printf("sorting %d data took %ld us\n",
	   n, us);

    check(output, n);
    /*print(data, n);*/
  }

  free(data);

  return 0;
}