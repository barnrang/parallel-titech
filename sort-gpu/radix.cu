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
const int FIND_MAX_THREADS = 512; //allocate to shared memory

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
        if (arr[i] > arr[i+1]) printf("arr[%d] > arr[%d] - %d > %d\n", i, i+1, arr[i], arr[i+1]);
    }
}

__global__
void  findMax(int* arr, int* collectMax, int N)
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
      collectMax[blockIdx.x] = s_inputVals[0];
    
    }

__global__ void  scanSB(int* arr, 
    int *collectScan,
    int *collectSumScan,
    int *sumBlock,
    int pos,
    int N,
    int compare,
    int numMaxBlock) 
{
    __shared__ int s_inputVals[FIND_MAX_THREADS];
    __shared__ int s_inputValsTMP[FIND_MAX_THREADS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        s_inputVals[threadIdx.x] = (arr[idx] & pos) == compare;
        // printf("%d ", s_inputVals[threadIdx.x]);
        collectScan[idx] = s_inputVals[threadIdx.x];
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
        
        if(idx < N) collectSumScan[idx] = s_inputVals[threadIdx.x];
        sumBlock[blockIdx.x] = s_inputVals[FIND_MAX_THREADS - 1];
    } else {
        
        if(idx < N) collectSumScan[idx] = s_inputValsTMP[threadIdx.x];
        sumBlock[blockIdx.x] = s_inputValsTMP[FIND_MAX_THREADS - 1];
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
        if(idx < numMaxBlock) d_sumBlock[idx + 1] = s_sumBlock[idx];
        //else d_sumBlock[0] = 0;
      } else {
        if(idx < numMaxBlock) d_sumBlock[idx + 1] = s_sumBlockTMP[idx];
        //else d_sumBlock[0] = 0;
      }
      
    }

__global__ void  mergeScan(int* arr,
    int* collectScan,
    int* collectSumScan,
    int* sumBlock,
    int* interVals,
    int offset,
    int N)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // if (idx == 0) printf("{%d} ", collectScan[idx]);
        if (collectScan[idx]==0 || idx >= N) return;
        interVals[collectSumScan[idx] + sumBlock[blockIdx.x] + offset - 1] = arr[idx];
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

long time_diff_us(struct timeval st, struct timeval et)
{
  return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
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
    int* arr;
    checkCudaErrors(cudaMallocManaged(&arr, sizeof(int) * N));
    // curandState *state;
    // cudaMallocManaged(&state, sizeof(curandState));
    // makeData<<<numMaxBlock, FIND_MAX_THREADS>>>(arr, N, state);
    // cudaDeviceSynchronize();


    make_data(arr, N);
    printf("Prepared data\n");
    // print_array(arr, N);
 
    int *collectSumScan, *interVals, *sumBlock;
    int *collectScan;
    checkCudaErrors(cudaMallocManaged(&collectSumScan, sizeof(int) * N));
    checkCudaErrors(cudaMallocManaged(&collectScan, sizeof(int) * N));
    checkCudaErrors(cudaMallocManaged(&interVals, sizeof(int) * N));
    checkCudaErrors(cudaMallocManaged(&sumBlock, sizeof(int) * (numMaxBlock+1)));
    int* collectMax, *arbitary;
    cudaMallocManaged(&collectMax, sizeof(int) * numMaxBlock);
    cudaMallocManaged(&arbitary, sizeof(int) * numMaxBlock);

    /* Search for Maximum */
    gettimeofday(&st, NULL);

    int MAX = 0;
    // for (int i = 0; i < numMaxBlock; i++){
    //     if (MAX < collectMax[i]) MAX = collectMax[i];
    // }
    findMax <<<numMaxBlock,FIND_MAX_THREADS>>>(arr, collectMax, N);
    cudaDeviceSynchronize();
    // print_array(collectMax, numMaxBlock);
    int num_groups = numMaxBlock;
    while (num_groups > FIND_MAX_THREADS) {
        num_groups = (num_groups + FIND_MAX_THREADS - 1) / FIND_MAX_THREADS;
        findMax<<<num_groups, FIND_MAX_THREADS>>>(collectMax, arbitary, num_groups);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpy(collectMax, arbitary, sizeof(int) * num_groups, cudaMemcpyDeviceToDevice));
    }
    findMax<<<1, num_groups>>>(collectMax, collectMax, numMaxBlock);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize();

    MAX = collectMax[0];
    
    int step = (int)log2(MAX) + 2;
    printf("max = %d\n", MAX);
    printf("Max found\n");

    /* Loop through each bit digit */
    int MSB = 1;
    for (int i = 0; i < step; i++) {
        // printf("hello");
        scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(arr, 
            collectScan,
            collectSumScan,
            sumBlock,
            MSB,
            N,
            0, 
            numMaxBlock);
        cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
        // print_array(collectScan, N);
        // print_array(collectSumScan, N);
        // printf("{%d}\n", collectScan[0]);
        // int num_group = numMaxBlock;
        // while (num_group < FIND_MAX_THREADS) {

        // }
        scanBlockSum<<<1, numMaxBlock>>>(sumBlock, numMaxBlock);
        // print_array(collectSumScan, N);
        cudaDeviceSynchronize();// checkCudaErrors(cudaGetLastError());
        sumBlock[0] = 0;
        mergeScan<<<numMaxBlock, FIND_MAX_THREADS>>>(arr,
            collectScan,
            collectSumScan,
            sumBlock,
            interVals,
            0,
            N);
        cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

        int offset = sumBlock[numMaxBlock];
        scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(arr, 
            collectScan,
            collectSumScan,
            sumBlock,
            MSB,
            N,
            MSB, 
            numMaxBlock);
        cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
        // print_array(collectScan, N);
        // print_array(collectSumScan, N);
        scanBlockSum<<<1, numMaxBlock>>>(sumBlock, numMaxBlock);
        cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
        sumBlock[0] = 0;
        // printf("%d\n", offset);
        mergeScan<<<numMaxBlock, FIND_MAX_THREADS>>>(arr,
            collectScan,
            collectSumScan,
            sumBlock,
            interVals,
            offset,
            N);
        cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
        // copyData<<<numMaxBlock, FIND_MAX_THREADS>>>(arr, interVals, N);
        checkCudaErrors(cudaMemcpy(arr, interVals, sizeof(int) * N, cudaMemcpyDeviceToDevice));
        // cudaDeviceSynchronize();
        // print_array(arr, N);
        // printf("Finish a lOOP\n");
        MSB *= 2;
    }

    gettimeofday(&et, NULL); /* get start time */

    long us = time_diff_us(st, et);

    printf("sorting %d data took %ld us\n",N, us);    

    // printf("Finish lOOP");
    // print_array(arr, N);
    check_sort(arr, N);

    checkCudaErrors(cudaFree(collectSumScan));
    checkCudaErrors(cudaFree(collectScan));
    checkCudaErrors(cudaFree(interVals));
    checkCudaErrors(cudaFree(sumBlock));
    checkCudaErrors(cudaFree(arr));
    // checkCudaErrors(cudaFree(state));

    

    
    return 0;
}