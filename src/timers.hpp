#pragma once
#include <cuda_runtime.h>

struct CudaTimer {
    cudaEvent_t a,b;
    CudaTimer(){ cudaEventCreate(&a); cudaEventCreate(&b); }
    ~CudaTimer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(){ cudaEventRecord(a); }
    float stop(){ cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms; }
};
