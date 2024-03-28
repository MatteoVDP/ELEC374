
//written by Matteo Van der Plaat (20287556)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MATRIX_WIDTH 100 //dimensions of matrices
#define MATRIX_SIZE (MATRIX_WIDTH * MATRIX_WIDTH) //total number of elements in matrices
#define NBYTES (MATRIX_SIZE * sizeof(float))

//int BLOCK_WIDTH = 1;
int BLOCK_WIDTH[] = { 2, 5, 10, 25, 32 };
int TILE_SIZE[] = { 2, 5, 10, 25 };
#define TILE_WIDTH 25 //tile width used in tiledmatrix multiplication

float M[MATRIX_SIZE];
float N[MATRIX_SIZE];
float P[MATRIX_SIZE];

//functions to be tested
void cudaTransferTest();
void cudaMatMult(float* M, float* N, float* P, int WIDTH);

//matrix multiplication kernel, called by cudaMatMult function
__global__ void tiledMatMultKernel(float* M, float* N, float* P, int WIDTH)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// calculate row, col index
	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;

	// Loop over the M and N tiles required to compute the P element
	for (int i = 0; i < WIDTH / TILE_WIDTH; ++i)
	{
		// Collaborative loading of M and N tiles into shared memory
		Mds[ty][tx] = M[row * WIDTH + i * TILE_WIDTH + tx];
		Nds[ty][tx] = N[(i * TILE_WIDTH + ty) * WIDTH + col];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) // Perform one phase of dot product
			Pvalue += Mds[ty][k] * Nds[k][tx];

		__syncthreads();

	}
	P[row * WIDTH + col] = Pvalue; // All threads write to their P element

}

//function used to check validity of value outputted by GPU Mat Mult function
void checkGPUanswer(float* M, float* N, float* GPU_P, int WIDTH)
{
	bool passed;
	float check;

	//calculate correct values in CPU and compare against GPU value
	for (int i = 0; i < WIDTH; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			check = 0;

			for (int k = 0; k < WIDTH; k++)
			{
				check += M[i * WIDTH + k] * N[k * WIDTH + j];
				if (GPU_P[i * WIDTH + j] != check) passed = 0;
			}
		}
	}
	passed = 1; //if all values match up, test passed

	if (passed)	printf("TEST PASSED\n");
	else		printf("TEST FAILED\n");
}

//standard matrix multiplication, computed using CPU
void CPUmatMult(float* M, float* N, float* P, int WIDTH)
{
	//initialization values used for timing
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	for (int l = 0; l < 5; l++)
	{
		cudaEventRecord(start, 0); // start timer
		cudaDeviceSynchronize();

		for (int i = 0; i < WIDTH; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				for (int k = 0; k < WIDTH; k++)
				{
					P[i * WIDTH + j] += M[i * WIDTH + k] * N[k * WIDTH + j];
				}
			}
		}

		cudaEventRecord(stop, 0);	// end timer
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("Time for CPU matrix multiplication: %f\n", gpu_time); //display results
	}


}

int main()
{
	srand(time(NULL)); //seed random function

	//allocate memory in host
	cudaMallocHost((void**)&M, NBYTES);
	cudaMallocHost((void**)&N, NBYTES);
	cudaMallocHost((void**)&P, NBYTES);

	for (int i = 0; i < MATRIX_SIZE; i++)
	{	// fill matrices M and N with randon values for testing
		M[i] = rand() % 100 / (float)10.0;
		N[i] = rand() % 100 / (float)10.0;
		P[i] = 0.0;
	}

	//function used for testing transferring data between host and device
	//cudaTransferTest();

	//functions used for testing matrix multiplication using GPUs and comparing against CPUs
	//cudaMatMult(M, N, P, MATRIX_WIDTH); // GPU/Cuda matrix multiplication
	//CPUmatMult(M, N, P, MATRIX_WIDTH); // CPU matrix multiplication

	cudaDeviceProp dev_prop;
	int dev_count;
	cudaGetDeviceCount(&dev_count);

	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&dev_prop, i);
		printf("mp count: %d\n", dev_prop.multiProcessorCount);
		printf("name: %s\n", dev_prop.name);
		printf("blocks per smp: %d\n", dev_prop.maxBlocksPerMultiProcessor);
		printf("max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
		printf("shared mem per MP: %d\n", dev_prop.sharedMemPerMultiprocessor);
		printf("max threads dimension 0: %d\n", dev_prop.maxThreadsDim[0]);
		printf("max threads dimension 1: %d\n", dev_prop.maxThreadsDim[1]);
		printf("max threads dimension 2: %d\n", dev_prop.maxThreadsDim[2]);
		printf("max grid size dimension 0: %d\n", dev_prop.maxGridSize[0]);
		printf("max grid size dimension 1: %d\n", dev_prop.maxGridSize[1]);
		printf("max grid size dimension 2: %d\n", dev_prop.maxGridSize[2]);
		printf("regs per block: %d\n", dev_prop.regsPerBlock);
		printf("shared mem per block: %lu\n", dev_prop.sharedMemPerBlock);

	}

	//free host memory 
	cudaFreeHost(M);
	cudaFreeHost(N);
	cudaFreeHost(P);

	return 0;
}

void cudaTransferTest()
{
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	float* dM, * dN, * dP;

	//allocate memory for matrices on device
	cudaMalloc((void**)(&dM), NBYTES);
	cudaMalloc((void**)(&dN), NBYTES);
	cudaMalloc((void**)(&dP), NBYTES);

	//check memory allocation was successful
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error allocating memory in device");

	//repeat 5 times to ensure correctness
	for (int i = 0; i < 5; i++)
	{
		cudaEventRecord(start, 0); // start timer
		cudaDeviceSynchronize();

		//copy information from host to device
		cudaMemcpy(dM, M, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dN, N, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventRecord(stop, 0); // end timer
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("Time to send matrices to device from host: %f\n", gpu_time); //display results

		cudaEventRecord(start, 0); // start timer
		cudaDeviceSynchronize();

		//copy information from device to host
		cudaMemcpy(M, dM, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(N, dN, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop, 0);	// end timer
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("Time to send matrices to host from device: %f\n", gpu_time); //display results


	}
	//free device memory
	cudaFree(dM);
	cudaFree(dN);
	cudaFree(dP);

}

void cudaMatMult(float* M, float* N, float* P, int WIDTH)
{
	cudaEvent_t start, stop;
	float gpu_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	float* dM, * dN, * dP;

	//allocate memory for matrices on device
	cudaMalloc((void**)(&dM), NBYTES);
	cudaMalloc((void**)(&dN), NBYTES);
	cudaMalloc((void**)(&dP), NBYTES);

	//check memory allocation was successful
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error allocating memory in device");

	//define dimensions of grid and blocks
	dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	for (int j = 0; j < 5; j++) //repeat trials 5 times to ensure accuracy
	{

		//copy memory from host to device
		cudaMemcpy(dM, M, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dN, N, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventRecord(start, 0); // start timer
		cudaDeviceSynchronize();

		//calculate matrix multiplication using Cuda and GPUs,, enabling synchronization
		tiledMatMultKernel << <dimGrid, dimBlock >> > (dM, dN, dP, WIDTH);

		cudaEventRecord(stop, 0); // end timer
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("Time for tiled GPU matrix multiplication: %f\n", gpu_time); //display results

		cudaMemcpy(M, dM, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(N, dN, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		//checkGPUanswer(M, N, P, MATRIX_WIDTH); //make sure answers are correct by comparing against CPU values

	}

	//free device memory
	cudaFree(dM);
	cudaFree(dN);
	cudaFree(dP);

}
