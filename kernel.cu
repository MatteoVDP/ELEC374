
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// number and type of cuda devices
// num cores
// amount of const mem

int getNumCores(cudaDeviceProp devprop)
{
    int mp = devprop.multiProcessorCount;

    switch (devprop.major)
    {
    case 2:
        return (devprop.minor == 1) ? (mp * 48) : (mp * 32);
    case 3: 
        return mp * 192;
    case 5: 
        return mp * 128;
    case 6:
        if (devprop.minor == 1 || devprop.minor == 2) return mp * 128;
        else if (devprop.minor == 0) return mp * 64;
        else return -1;
    case 7:
        if (devprop.minor == 0 || devprop.minor == 5) return mp * 64;
        else return -1;
    default:
        return -1;
    }
}

int main()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("number of devices: %d\n", dev_count);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        printf("max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("mp count: %d\n", dev_prop.multiProcessorCount);
        printf("clock rate: %d\n", dev_prop.clockRate);
        printf("warp size: %d\n", dev_prop.warpSize);
        printf("regs per block: %d\n", dev_prop.regsPerBlock);
        printf("shared mem per block: %lu\n", dev_prop.sharedMemPerBlock);
        printf("total global memory: %lu\n", dev_prop.totalGlobalMem);
        printf("total constant memory: %lu\n", dev_prop.totalConstMem);
        printf("max threads dimension 0: %d\n", dev_prop.maxThreadsDim[0]);
        printf("max threads dimension 1: %d\n", dev_prop.maxThreadsDim[1]);
        printf("max threads dimension 2: %d\n", dev_prop.maxThreadsDim[2]);
        printf("max grid size dimension 0: %d\n", dev_prop.maxGridSize[0]);
        printf("max grid size dimension 1: %d\n", dev_prop.maxGridSize[1]);
        printf("max grid size dimension 2: %d\n", dev_prop.maxGridSize[2]);
        printf("name: %s\n", dev_prop.name);
        printf("name: %d\n", getNumCores(dev_prop));

    }
}



/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

*/