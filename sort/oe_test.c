#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int n=100000000;
int num_threads=4;

int init(double *data, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    data[i] = (double)rand() / RAND_MAX;
  }
  return 0;
}

int main(int argc, char* argv[]) 
{
    int i,j;
    float temp;
    double* a;
    a = malloc(sizeof(double) * n);

    init(a, n);

    num_threads = atol(argv[1]);

    double start_time;
    start_time = omp_get_wtime();

    
    for(i=0;i<1;i++)
    {    
        if(i%2==0)
        {
            #pragma omp parallel for private(temp,j) shared(a) num_threads(num_threads)
            for(j=0;j<n-1;j+=2)
            {
                // printf("\n%d", j);
                if(a[j]> a[j+1])
                {
                    temp = a[j];
                    a[j] = a[j+1];
                    a[j+1] = temp;
                }
            }
        }
        else
        {
            #pragma omp parallel for private(temp,j) shared(a) num_threads(num_threads)
            for(j=1;j<n-1;j+=2)
            {
                if(a[j]> a[j+1])
                {
                    temp = a[j];
                    a[j] = a[j+1];
                    a[j+1] = temp;
                }
            }
        }
        
    }

    printf("\n Execution time = %lf seconds\n", omp_get_wtime() - start_time);
}