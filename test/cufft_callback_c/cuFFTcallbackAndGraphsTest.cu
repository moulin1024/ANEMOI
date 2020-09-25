//
// Forward and backwards cuFFT Benchmar using graphs and callbacks.
//
// This test case measures the runtime of a set of function calls with the following order:
//   -> work on the real array (init kernel) 
//   -> forward FFT 
//   -> work on the complex array (work kernel)
//   -> backwards FFT 
//   -> work on the real array (post kernel)
//
// This benchmark compares the execution of the calls above using different methods
// available in CUDA. These are:
//   -> Regular independent calls on the same stream.
//   -> Packing all the calls into a CUDA graph.
//   -> Fusing the non-FFT kernels into the cuFFT calls with callbacks.
//   -> Packing the previous method into a graph.
//

#include <iostream>
#include <cufft.h>
#include <cufftXt.h>

// This array layout is equivalent to a 2D array with the transforms done in the innermost dimension.
#define INPUT_SIGNAL_SIZE 128
#define BATCH_SIZE 20000
#define COMPLEX_SIGNAL_SIZE (INPUT_SIGNAL_SIZE/2 + 1)

#define BLOCK_SIZE 128
#define REPEAT 1000

// Regular kernels
__global__ void init(double * arr, int n)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (id < n) 
  {
    arr[id] = cos( ((double)id) / ((double)INPUT_SIGNAL_SIZE) * 10.0f * 3.141592f );
  }
}

__global__ void work (cufftDoubleComplex * arr, int n)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if ( id < n ) 
  {
    arr[id] = cufftDoubleComplex{2.0,2.0};
  }
}

__global__ void post(double * arr, int n)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (id < n) 
  {
    arr[id] = arr[id] * arr[id] - 0.5f;
  }
}

// Device callbacks
__device__ double CB_init(void * dataIn, size_t offset, void * callerInfo, void * sharedPtr)
{
  return cos( ((double)offset) / ((double)INPUT_SIGNAL_SIZE) * 10.0f * 3.141592f );
}

__device__ void CB_work(void * dataOut, size_t offset, cufftDoubleComplex element, void * callerInfo, void * sharedPtr)
{
  ((cufftDoubleComplex*)dataOut)[offset] = cufftDoubleComplex{2.0,2.0};
}

__device__ void CB_post(void * dataOut, size_t offset, cufftDoubleReal element, void * callerInfo, void * sharedPtr)
{
  ((cufftDoubleReal*)dataOut)[offset] = element * element - 0.5;
}

__managed__ cufftCallbackLoadD d_loadCallbackInit = CB_init;
__managed__ cufftCallbackStoreZ d_storeCallbackWork = CB_work;
__managed__ cufftCallbackStoreD d_storeCallbackPost = CB_post;

int main()
{

  cufftDoubleReal * in_d;
  cufftDoubleComplex * out_d; 
 
  cudaMalloc((void**)&in_d,  sizeof(cufftDoubleReal)*INPUT_SIGNAL_SIZE*BATCH_SIZE);
  cudaMalloc((void**)&out_d, sizeof(cufftDoubleComplex)*COMPLEX_SIGNAL_SIZE*BATCH_SIZE);
 
  cufftHandle planForward, planBackwards;
  int transformSizeIn = INPUT_SIGNAL_SIZE;
  int transformSizeOut = COMPLEX_SIGNAL_SIZE;
  cufftPlanMany(&planForward, 1, &transformSizeIn, &transformSizeIn, 1, transformSizeIn, 
                                                   &transformSizeOut, 1, transformSizeOut,
                                                   CUFFT_D2Z, BATCH_SIZE);
  cufftPlanMany(&planBackwards, 1, &transformSizeOut, &transformSizeOut, 1, transformSizeOut, 
                                                      &transformSizeIn, 1, transformSizeIn,
                                                      CUFFT_Z2D, BATCH_SIZE);
  // Attach plans to a cuda stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cufftSetStream(planForward, stream);
  cufftSetStream(planBackwards, stream);

  // Declare event variables for timing purposes
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*
    Regular independent call of the kernels and cufft functions.
  */
  cudaEventRecord(start, 0);
  for ( int i = 0; i < REPEAT; i++)
  {
    // Initialize the input data.
    init<<<(INPUT_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(in_d, INPUT_SIGNAL_SIZE);
 
    // Forwards & backwards FFTs 
    cufftExecD2Z(planForward, in_d, out_d); 
    work<<<(COMPLEX_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(out_d, COMPLEX_SIGNAL_SIZE);
    cufftExecZ2D(planBackwards, out_d, in_d); 
    post<<<(INPUT_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(in_d, INPUT_SIGNAL_SIZE);
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime, baseline;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  baseline = elapsedTime/(float)REPEAT;
  std::cout << "Elapsed time of the regular cufft was " << baseline << " ms" << std::endl;

  /* 
    Graph-based FFT launches
  */
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Capure the graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  // Same work as with the loop above.
  init<<<(INPUT_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(in_d, INPUT_SIGNAL_SIZE);
  cufftExecD2Z(planForward, in_d, out_d); 
  work<<<(COMPLEX_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(out_d, COMPLEX_SIGNAL_SIZE);
  cufftExecZ2D(planBackwards, out_d, in_d); 
  post<<<(INPUT_SIGNAL_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(in_d, INPUT_SIGNAL_SIZE);
  
  // End stream Capture & instantiate the graph
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

  // Repeat the work done above, but using graaphs this time
  cudaEventRecord(start, 0);
  for ( int i = 0; i < REPEAT; i++)
  {
    cudaGraphLaunch(instance, stream); 
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Speed-up from using CUDA graphs: " << baseline/(elapsedTime/(float)REPEAT) << "X" << std::endl;

  /* 
    Same as above, but with callbacks
  */
  cufftXtSetCallback(planForward, (void**)&d_loadCallbackInit, CUFFT_CB_LD_REAL_DOUBLE, 0); 
  cufftXtSetCallback(planForward, (void**)&d_storeCallbackWork, CUFFT_CB_ST_COMPLEX_DOUBLE, 0); 
  cufftXtSetCallback(planBackwards, (void**)&d_storeCallbackPost, CUFFT_CB_ST_REAL_DOUBLE, 0); 

  cudaEventRecord(start, 0);
  for ( int i = 0; i < REPEAT; i++)
  {
    // Forwards & backwards FFTs 
    cufftExecD2Z(planForward, in_d, out_d); 
    cufftExecZ2D(planBackwards, out_d, in_d); 
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Speed-up form using cuFFT callbacks: " << baseline/(elapsedTime/(float)REPEAT) << "X" << std::endl;

  /* 
    Graph-based FFTs with callbacks 
  */
  cudaGraph_t graph2;
  cudaGraphExec_t instance2;

  // Capure the graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  // Same work as with the loop above.
  cufftExecD2Z(planForward, in_d, out_d); 
  cufftExecZ2D(planBackwards, out_d, in_d); 
  
  // End stream Capture & instantiate the graph
  cudaStreamEndCapture(stream, &graph2);
  cudaGraphInstantiate(&instance2, graph2, NULL, NULL, 0);

  // Repeat the work done above, but using graaphs this time
  cudaEventRecord(start, 0);
  for ( int i = 0; i < REPEAT; i++)
  {
    cudaGraphLaunch(instance2, stream); 
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Speed-up from using cuFFT callbacks and CUDA graphs: " << baseline/(elapsedTime/(float)REPEAT) << "X" << std::endl;

  // Free the memory
  cufftDestroy(planForward);
  cufftDestroy(planBackwards);
  cudaFree(in_d); 
  cudaFree(out_d); 

  return 0;

}
