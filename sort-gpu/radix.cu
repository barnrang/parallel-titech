#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <cuda.h>
// #include <curand_kernel.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <helper_cuda.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// #include <thrust/host_vector.h>

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

const int block_size = 1024;
const int DIM = 32;
const int MAX_THREADS_PER_BLOCK = 65535;
const int FIND_MAX_THREADS = 1024; //allocate to shared memory

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
        arr[i] = rand_lim(4095);
    }
    printf("Finish making data\n");
}

void check_sort(int* arr, int N){
    for (int i = 0; i < N-1; i++) {
        // printf("%d\n", arr[i]);
        if (arr[i] > arr[i+1]) fprintf(stderr, "arr[%d] > arr[%d] - %d > %d\n", i, i+1, arr[i], arr[i+1]);
    }
    fflush(stderr);
}

__global__
void  findMax(int* arr, int* d_collectMax, int N)
    {
      __shared__ int s_inputVals[FIND_MAX_THREADS];
    //   printf("Enter kernelll");
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N){ s_inputVals[threadIdx.x] = arr[idx];
        // printf("%d\n", arr[idx]);    
    }
      else s_inputVals[threadIdx.x] = 0;
      __syncthreads();
    
      int half = FIND_MAX_THREADS / 2;
      while (half != 0) {
        if (threadIdx.x < half) {
          s_inputVals[threadIdx.x] = max(s_inputVals[threadIdx.x], s_inputVals[threadIdx.x + half]);
        }
        half /= 2;
        __syncthreads();
      }
      d_collectMax[blockIdx.x] = s_inputVals[0];
    
    }
    

__global__ void  markArr(int* d_arr, 
    int *d_collectScan,
    int pos,
    int N,
    int compare) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        d_collectScan[idx] = (d_arr[idx] & pos) == compare;
    } 
}

__global__ void  scanSB(int *d_collectScan,
    int *d_collectSumScan,
    int *d_sumBlock,
    int N,
    int numMaxBlock) 
{
    __shared__ int s_inputVals[FIND_MAX_THREADS];
    __shared__ int s_inputValsTMP[FIND_MAX_THREADS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        s_inputVals[threadIdx.x] = d_collectScan[idx];
    } else s_inputVals[threadIdx.x] = 0;
    __syncthreads();
    
    int dist = 1;
    int count = 0;
    
    
    while (dist < FIND_MAX_THREADS) {
        if (count % 2 == 0){
        s_inputValsTMP[threadIdx.x] = s_inputVals[threadIdx.x];
        if (threadIdx.x >= dist) {
            s_inputValsTMP[threadIdx.x] += s_inputVals[threadIdx.x - dist];
        }
        }
        else {
        s_inputVals[threadIdx.x] = s_inputValsTMP[threadIdx.x];
        if (threadIdx.x >= dist) {
            s_inputVals[threadIdx.x] += s_inputValsTMP[threadIdx.x - dist];
        }
        }
        dist *= 2;
        count++;
        __syncthreads(); 
    }
    if (count % 2 == 0){
        
        if(idx < N) d_collectSumScan[idx] = s_inputVals[threadIdx.x];
        d_sumBlock[blockIdx.x] = s_inputVals[FIND_MAX_THREADS - 1];
    } else {
        
        if(idx < N) d_collectSumScan[idx] = s_inputValsTMP[threadIdx.x];
        d_sumBlock[blockIdx.x] = s_inputValsTMP[FIND_MAX_THREADS - 1];
    }
    
}

__global__ void  scanBlockSum(int *d_sumBlock,
    int numMaxBlock)
    {
      __shared__ int s_sumBlock[FIND_MAX_THREADS];
      __shared__ int s_sumBlockTMP[FIND_MAX_THREADS];
      int idx = threadIdx.x;
      if(idx >= numMaxBlock) return;
      s_sumBlock[idx] = d_sumBlock[idx];
      __syncthreads();
    
      int dist = 1;
      int count = 0;
      while (dist < numMaxBlock) {
        if(count % 2 == 0){
          s_sumBlockTMP[idx] = s_sumBlock[idx];
          if (idx >= dist) {
            s_sumBlockTMP[idx] += s_sumBlock[idx - dist];
          }
        }
        else {
          s_sumBlock[idx] = s_sumBlockTMP[idx];
          if (idx >= dist) {
            s_sumBlock[idx] += s_sumBlockTMP[idx - dist];
          }
        }
        
        dist *= 2;
        count++;
        __syncthreads();
      }
      if (count % 2 == 0){
        if(idx < numMaxBlock) d_sumBlock[idx] = s_sumBlock[idx];
        //else d_sumBlock[0] = 0;
      } else {
        if(idx < numMaxBlock) d_sumBlock[idx] = s_sumBlockTMP[idx];
        //else d_sumBlock[0] = 0;
      }

    //   d_sumBlock[0] = 0;
      
    }

__global__ void  mergeScan(int* arr,
    int* d_collectScan,
    int* d_collectSumScan,
    int* d_interVals,
    int offset,
    int N)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // if (idx == 0) printf("{%d} ", d_collectScan[idx]);
        if (d_collectScan[idx]==0 || idx >= N) return;
        d_interVals[d_collectSumScan[idx] + offset - 1] = arr[idx];
    }


__global__ void  copyData(int* d_dst, 
    int* d_src, 
    int N)
    {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;
    d_dst[idx] = d_src[idx];
    }

// __global__ void makeData(int* arr,
//     int N,
//     curandState* state) {
//         int idx = threadIdx.x + blockIdx.x * blockDim.x;
//         float myrandf = curand_uniform(state + idx);
//         myrandf *= (255 + 0.999999);
//         arr[idx]= (int)truncf(myrandf);
//     }

void print_array(int* arr, int N){
    for(int i = 0; i < N; i++) {
        if (i == 0){
            printf("[%d, ", arr[i]);
        } else if (i == N -1 ) {
            printf("%d]\n", arr[i]);
        } else {
            printf("%d, ", arr[i]);
        }
    }
}

void print_darray(int* d_arr, int N){
    int *arr; cudaMallocHost(&arr, N * sizeof(int));
    checkCudaErrors(cudaMemcpy(arr, d_arr, sizeof(int) * N, cudaMemcpyDeviceToHost));
    print_array(arr, N);
    cudaFree(arr);
}

long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}


__global__
void mergeScanToIndex(
    int* d_toCollect,
    int* d_sumBlock,
    int N
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;
    if (blockIdx.x > 0) d_toCollect[idx] = d_sumBlock[blockIdx.x-1] + d_toCollect[idx];
}

void recursive_scan(
    int* d_toScan,
    int* d_toCollect,
    int N
)
{   
    int numBlockSize = (N + FIND_MAX_THREADS - 1) / FIND_MAX_THREADS;
    int* d_sumBlock;


    // For debug
    // printf("before N = %d ", N);
    // print_darray(d_toScan, N);
    
    
    checkCudaErrors(cudaMalloc(&d_sumBlock, sizeof(int) * (numBlockSize + 1)));
    scanSB<<<numBlockSize, FIND_MAX_THREADS>>>(
        d_toScan,
        d_toCollect,
        d_sumBlock,
        N,
        numBlockSize
    );
    
    cudaDeviceSynchronize();
    // printf("mid N = %d ", N);
    // print_darray(d_sumBlock, numBlockSize+1);
    // printf("scan N = %d\n", N);
    if (numBlockSize > FIND_MAX_THREADS) {
        recursive_scan(
            d_sumBlock,
            d_sumBlock,
            numBlockSize
        );
        mergeScanToIndex<<<numBlockSize, FIND_MAX_THREADS>>>(d_toCollect, d_sumBlock, N);
        cudaDeviceSynchronize();
    } else {
        scanBlockSum<<<1, FIND_MAX_THREADS>>>(d_sumBlock, numBlockSize);
        // printf("mid2 N = %d ", N);
        // print_darray(d_sumBlock, numBlockSize+1);
        cudaDeviceSynchronize();
        // printf("merge N = %d\n", N);
        mergeScanToIndex<<<numBlockSize, FIND_MAX_THREADS>>>(d_toCollect, d_sumBlock, N);
        cudaDeviceSynchronize();
        // printf("merge2 N = %d\n", N);
    }
    checkCudaErrors(cudaFree(d_sumBlock));
    // printf("After N = %d ", N);
    // print_darray(d_toCollect, N);

}

void scanAndMerge(int* d_arr,
    int* d_collectScan,
    int* d_collectSumScan,
    int* d_sumBlock,
    int* d_interVals,
    int MSB,
    int N,
    int compare,
    int numMaxBlock,
    int offset)
{

    markArr<<<numMaxBlock, FIND_MAX_THREADS>>>(d_arr, 
        d_collectScan,
        MSB,
        N,
        compare) ;

        cudaDeviceSynchronize();
        // checkCudaErrors(cudaGetLastError());

    // printf("Mark!!\n");
    recursive_scan(
        d_collectScan,
        d_collectSumScan,
        N
    );
    // printf("Scan!!\n");

    // scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(d_collectScan,
    //     d_collectSumScan,
    //     d_sumBlock,
    //     N,
    //     numMaxBlock);

    // scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(d_arr, 
    //     d_collectScan,
    //     d_collectSumScan,
    //     d_sumBlock,
    //     MSB,
    //     N,
    //     compare, 
    //     numMaxBlock);
    // cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
    // print_d_array(d_collectScan, N);
    // print_d_array(d_collectSumScan, N);
    // scanBlockSum<<<1, numMaxBlock>>>(d_sumBlock, numMaxBlock);
    // cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
    // d_sumBlock[0] = 0;
    // checkCudaErrors(cudaMemset(d_sumBlock, 0, sizeof(int)));
    // printf("%d\n", offset);
    mergeScan<<<numMaxBlock, FIND_MAX_THREADS>>>(d_arr,
        d_collectScan,
        d_collectSumScan,
        d_interVals,
        offset,
        N);
    cudaDeviceSynchronize();
}


int main(int argc, char *argv[]) {

    struct timeval st;
    struct timeval et;

    srand (time(0));
    int N = 1000000;

    if (argc >= 2) {
        N = atol(argv[1]);
    }

    int numMaxBlock = (N + FIND_MAX_THREADS - 1) / FIND_MAX_THREADS;
    int *d_arr, *arr;
    checkCudaErrors(cudaMallocHost(&arr, sizeof(int) * N));
    make_data(arr, N);
    // print_array(arr, N);

    checkCudaErrors(cudaMalloc(&d_arr, sizeof(int) * N));
    checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int) * N, cudaMemcpyHostToDevice));

    // curandState *state;
    // cudaMalloc(&state, sizeof(curandState));
    // makeData<<<numMaxBlock, FIND_MAX_THREADS>>>(arr, N, state);
    // cudaDeviceSynchronize();


    
    printf("Prepared data\n");
    // print_array(arr, N);
 
    int *d_collectSumScan, *d_interVals, *d_sumBlock;
    int *d_collectScan;
    checkCudaErrors(cudaMalloc(&d_collectSumScan, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&d_collectScan, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&d_interVals, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&d_sumBlock, sizeof(int) * (numMaxBlock+1)));
    int* d_collectMax, *d_arbitary;
    cudaMalloc(&d_collectMax, sizeof(int) * numMaxBlock);
    cudaMalloc(&d_arbitary, sizeof(int) * numMaxBlock);

    /* Search for Maximum */
    gettimeofday(&st, NULL);

    int MAX = 0;
    
    findMax <<<numMaxBlock,FIND_MAX_THREADS>>>(arr, d_collectMax, N);
    cudaDeviceSynchronize();
    // for (int i = 0; i < numMaxBlock; i++){
    //     if (MAX < d_collectMax[i]) MAX = d_collectMax[i];
    // }
    // print_array(d_collectMax, numMaxBlock);
    int num_groups = numMaxBlock;
    while (num_groups > FIND_MAX_THREADS) {
        num_groups = (num_groups + FIND_MAX_THREADS - 1) / FIND_MAX_THREADS;
        findMax<<<num_groups, FIND_MAX_THREADS>>>(d_collectMax, d_arbitary, num_groups);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpy(d_collectMax, d_arbitary, sizeof(int) * num_groups, cudaMemcpyDeviceToDevice));
    }
    findMax<<<1, num_groups>>>(d_collectMax, d_collectMax, numMaxBlock);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&MAX, &d_collectMax[0], sizeof(int), cudaMemcpyDeviceToHost));
    
    int step = (int)log2(MAX) + 1;
    printf("max = %d\n", MAX);
    printf("Max found\n");
    printf("num Max block = %d\n", numMaxBlock);
    // printf("hello11\n");

    /* Loop through each bit digit */
    int MSB = 1;
    int i = 0;
    int offset;
    for (i = 0; i < step; i++) {
        // printf("hello\n");
        if (i % 2 == 0){
            scanAndMerge(d_arr,
                d_collectScan,
                d_collectSumScan,
                d_sumBlock,
                d_interVals,
                MSB,
                N,
                0,
                numMaxBlock,
                0);
                
                // printf("hello2\n");
            checkCudaErrors(cudaMemcpy(&offset, &d_collectSumScan[N-1], sizeof(int), cudaMemcpyDeviceToHost));
            // printf("offset = %d\n", offset);
            // printf("hello3\n");
            scanAndMerge(d_arr,
                d_collectScan,
                d_collectSumScan,
                d_sumBlock,
                d_interVals,
                MSB,
                N,
                MSB,
                numMaxBlock,
                offset);
        } else {
            scanAndMerge(d_interVals,
                d_collectScan,
                d_collectSumScan,
                d_sumBlock,
                d_arr,
                MSB,
                N,
                0,
                numMaxBlock,
                0);
    
            // int offset = d_sumBlock[numMaxBlock];
            // checkCudaErrors(cudaMemcpy(&offset, &d_sumBlock[numMaxBlock], sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&offset, &d_collectSumScan[N-1], sizeof(int), cudaMemcpyDeviceToHost));
            // printf("offset = %d\n", offset);

            scanAndMerge(d_interVals,
                d_collectScan,
                d_collectSumScan,
                d_sumBlock,
                d_arr,
                MSB,
                N,
                MSB,
                numMaxBlock,
                offset);
        }
        MSB *= 2;
        // if (i % 2==0){
        //     checkCudaErrors(cudaMemcpy(arr, d_interVals, sizeof(int) * N, cudaMemcpyDeviceToHost));
        // } else {
        //     checkCudaErrors(cudaMemcpy(arr, d_arr, sizeof(int) * N, cudaMemcpyDeviceToHost));
        // }
        // print_array(arr, N);
    }

    if (i % 2!=0){
        checkCudaErrors(cudaMemcpy(arr, d_interVals, sizeof(int) * N, cudaMemcpyDeviceToHost));
    } else {
        checkCudaErrors(cudaMemcpy(arr, d_arr, sizeof(int) * N, cudaMemcpyDeviceToHost));
    }

    gettimeofday(&et, NULL); /* get start time */

    long us = time_diff_us(st, et);

    printf("sorting %d data took %ld us\n",N, us);    

    // printf("Finish lOOP");
    // print_array(arr, N);
    check_sort(arr, N);
    // print_array(arr, N);

    checkCudaErrors(cudaFree(d_collectSumScan));
    checkCudaErrors(cudaFree(d_collectScan));
    checkCudaErrors(cudaFree(d_interVals));
    checkCudaErrors(cudaFree(d_sumBlock));
    checkCudaErrors(cudaFree(d_arr));
    // checkCudaErrors(cudaFree(state));

    

    
    return 0;
}