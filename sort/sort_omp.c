#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

int print(double *data, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    printf("%.4lf ", data[i]);
  }
  printf("\n");
  return 0;
}

/* sort data[s] ... data[e-1] */
int sort(double *data, int s, int e, int thresh)
{
  int i, j;
  double pivot;

  if (e-s <= 1) {
    /* do nothing */
    return 0;
  }

  /* [Pivot selection] */
  /* select a pivot, one of data[s] ... data[e-1]  */
  /* pivot must not be the minimum value */
  pivot = data[s];
  for (i = s+1; i < e; i++) {
    if (data[i] > pivot) {
      pivot = data[i];
      break;
    }
    else if (data[i] < pivot) {
      break;
    }
  }

  if (i == e) {
    /* all values are same. do nothing */
    return 0;
  }

  /* [Partition data[] into 2 parts] */
  i = s;
  j = e-1;
  while (1) {
    double tmp;
    while (data[i] < pivot)
      i++;

    while (data[j] >= pivot)
      j--;

    if (i >= j) {
      /* partitioning finished */
      break;
    }

    /* swap */
    tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
    i++;
    j--;
  }
  /* Here, "i" is boundary of 2 parts */

  /* call sort() recursively */
  

  if ((e - s) > thresh){
    
    #pragma omp task
    {sort(data, s, i, thresh);} /* sort data[s] ... data[i-1] */
    #pragma omp task
    {sort(data, i, e, thresh);}  /* sort data[i] ... data[e-1] */
    
    } else {
      sort(data, s, i, thresh); /* sort data[s] ... data[i-1] */
      sort(data, i, e, thresh); /* sort data[i] ... data[e-1] */
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
  int i;
  int num_threads;
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

  int T = 10;
  long time_sum = 0;

  for (i = 0; i < T+1; i++) {
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
        sort(data, 0, n, thresh);
      }
    }
    gettimeofday(&et, NULL); /* get start time */
    if (i != 0){
      us = time_diff_us(st, et);
      printf("sorting %d data took %ld us\n",
      n, us);
      time_sum += us;
    }

    check(data, n);
    /*print(data, n);*/
  }

  printf("Average time took %ld\n", time_sum / T);

  free(data);

  return 0;
}
