
//written by Matteo Van der Plaat (20287556)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


//use attributes major and minor to determine number of cores per device
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
    cudaGetDeviceCount(&dev_count); //determine number of devices
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
        printf("number of cores: %d\n", getNumCores(dev_prop));

    }
}

