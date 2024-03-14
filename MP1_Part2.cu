
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MATRIX_WIDTH 1000 //dimensions of matrices
#define MATRIX_SIZE (MATRIX_WIDTH * MATRIX_WIDTH) //total number of elements in matrices
#define NBYTES (MATRIX_SIZE * sizeof(float))

//int BLOCK_WIDTH = 1;
int BLOCK_WIDTH[] = { 2, 5, 10, 25, 32 };

float M[MATRIX_SIZE];
float N[MATRIX_SIZE];
float P[MATRIX_SIZE];

//functions to be tested
void cudaTransferTest();
void cudaMatMult(float* M, float* N, float* P, int WIDTH);

//matrix multiplication kernel, called by cudaMatMult function
__global__ void matMultKernel(float* M, float* N, float* P, int WIDTH)
{
	// calculate row, col index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < WIDTH && col < WIDTH)
	{
		float Pvalue = 0;
		//each thread computes one element of the block sub-matrix
		for (int k = 0; k < WIDTH; k++) {
			Pvalue += M[row * WIDTH + k] * N[k * WIDTH + col];
		}
		P[row * WIDTH + col] = Pvalue;
	}
}

void checkGPUanswer(float* M, float* N, float* GPU_P, int WIDTH)
{
	bool passed;
	float check;

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
	passed = 1;

	if (passed)	printf("TEST PASSED\n");
	else		printf("TEST FAILED\n");
}

//standard matrix multiplication, computed using CPU
void CPUmatMult(float* M, float* N, float* P, int WIDTH)
{
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
	srand(time(NULL));

	cudaMallocHost((void**)&M, NBYTES);
	cudaMallocHost((void**)&N, NBYTES);
	cudaMallocHost((void**)&P, NBYTES);

	for (int i = 0; i < MATRIX_SIZE; i++)
	{	// value between 0 and 10, one decimal place
		M[i] = rand() % 100 / (float) 10.0;
		N[i] = rand() % 100 / (float) 10.0;
		P[i] = 0.0;
	}

	//cudaTransferTest();

	//for(int i = 0; i < 5; i++)
	//{
		cudaMatMult(M, N, P, MATRIX_WIDTH);
	//	CPUmatMult(M, N, P, MATRIX_WIDTH);
	//}

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

	for(int i = 0; i < 5; i++)
	{
		cudaEventRecord(start, 0); // start timer
		cudaDeviceSynchronize();

		//copy information from host to device
		cudaMemcpy(dM, M, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dN, N, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		cudaEventRecord(stop, 0);	// end timer
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

	cudaMemcpy(dM, M, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dN, N, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	for(int i = 0; i < 5; i++)
	{

		int NUM_BLOCKS = WIDTH / BLOCK_WIDTH[i];
		if (WIDTH % BLOCK_WIDTH[i]) NUM_BLOCKS++;

		dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS);
		dim3 dimBlock(BLOCK_WIDTH[i], BLOCK_WIDTH[i]);

		for (int i = 0; i < 5; i++)
		{
			cudaEventRecord(start, 0); // start timer
			cudaDeviceSynchronize();

			//copy information from device to host
			matMultKernel <<<dimGrid, dimBlock >>> (dM, dN, dP, WIDTH);

			cudaEventRecord(stop, 0);	// end timer
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpu_time, start, stop);
			printf("Time for GPU matrix multiplication: %f\n", gpu_time); //display results
			checkGPUanswer(M, N, P, MATRIX_WIDTH);

		}
	}

	cudaFree(dM);
	cudaFree(dN);
	cudaFree(dP);

}
