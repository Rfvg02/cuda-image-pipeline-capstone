#pragma once
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess){ \
  printf("CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  return; }} while(0)
#endif

// Convolución 1D horizontal con kernel gaussiano (odd: 3,5,7,9,...)
// Cada hilo procesa 1 píxel; borde con clamp
__global__ void gauss1d_h(const unsigned char* __restrict__ in, float* __restrict__ out,
                          int w, int h, const float* __restrict__ k, int r) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    float sum = 0.f;
    for (int dx=-r; dx<=r; ++dx) {
        int xx = x+dx; if (xx<0) xx=0; else if (xx>=w) xx=w-1;
        sum += k[dx+r] * (float)in[y*w + xx];
    }
    out[y*w + x] = sum;
}

// Convolución 1D vertical
__global__ void gauss1d_v(const float* __restrict__ in, unsigned char* __restrict__ out,
                          int w, int h, const float* __restrict__ k, int r) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    float sum = 0.f;
    for (int dy=-r; dy<=r; ++dy) {
        int yy = y+dy; if (yy<0) yy=0; else if (yy>=h) yy=h-1;
        sum += k[dy+r] * in[yy*w + x];
    }
    // clamp
    int p = (int)(sum + 0.5f);
    if (p<0) p=0; if (p>255) p=255;
    out[y*w + x] = (unsigned char)p;
}

// Sobel magnitud aproximada |Gx|+|Gy|
__global__ void sobel_mag(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                          int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;

    auto at=[&](int xx,int yy){
        if (xx<0) xx=0; else if (xx>=w) xx=w-1;
        if (yy<0) yy=0; else if (yy>=h) yy=h-1;
        return (int)in[yy*w + xx];
    };

    int gx = -at(x-1,y-1) + at(x+1,y-1)
             -2*at(x-1,y  ) + 2*at(x+1,y  )
             -at(x-1,y+1) + at(x+1,y+1);

    int gy = -at(x-1,y-1) -2*at(x,y-1) -at(x+1,y-1)
             +at(x-1,y+1) +2*at(x,y+1) +at(x+1,y+1);

    int mag = abs(gx) + abs(gy);
    if (mag>255) mag=255;
    out[y*w + x] = (unsigned char)mag;
}

// Construye kernel gaussiano 1D (host)
inline void build_gauss_kernel(float* k, int r, float sigma) {
    const int n = 2*r+1;
    float sum=0.f;
    for (int i=0;i<n;++i){
        int x=i-r;
        k[i]=expf(-(x*x)/(2.f*sigma*sigma));
        sum+=k[i];
    }
    for (int i=0;i<n;++i) k[i]/=sum;
}
