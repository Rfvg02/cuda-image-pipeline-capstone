#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "utils.hpp"
#include "kernels.cuh"
#include "timers.hpp"

static void die(const char* msg){ fprintf(stderr, "%s\n", msg); exit(1); }

int main(int argc, char** argv){
    // Parámetros
    int W=1920, H=1080, N=16;     // “decenas de grandes” por defecto
    int streams=4;
    int tpb=256;
    int tile=16; // no lo usamos intensivamente aquí, pero te deja jugar luego
    float sigma=1.5f;
    // parse rápido
    for (int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&](){ if(i+1>=argc) die("missing value"); return std::string(argv[++i]); };
        if (a=="--w")   W=std::stoi(next());
        else if (a=="--h") H=std::stoi(next());
        else if (a=="--n") N=std::stoi(next());
        else if (a=="--streams") streams=std::stoi(next());
        else if (a=="--tpb") tpb=std::stoi(next());
        else if (a=="--tile") tile=std::stoi(next());
        else if (a=="--sigma") sigma=std::stof(next());
    }

    printf("Config: %dx%d, images=%d, streams=%d, tpb=%d, sigma=%.2f\n",
           W,H,N,streams,tpb,sigma);

    // Directorios de salida
    ensure_dir("out");
    ensure_dir("out/results");

    // Construir kernel gaussiano (host)
    const int R=3; // radio 3 -> 7 taps
    std::vector<float> hK(2*R+1);
    build_gauss_kernel(hK.data(), R, sigma);

    // Reservas device compartidas por stream
    std::vector<cudaStream_t> S(streams);
    for (int i=0;i<streams;i++) cudaStreamCreate(&S[i]);

    // kernel 1D → copia a device
    float* dK=nullptr;
    cudaMalloc(&dK, sizeof(float)*(2*R+1));
    cudaMemcpy(dK, hK.data(), sizeof(float)*(2*R+1), cudaMemcpyHostToDevice);

    // buffers por stream
    std::vector<unsigned char*> dIn(streams,nullptr), dBlur(streams,nullptr), dEdge(streams,nullptr);
    std::vector<float*> dTmp(streams,nullptr);
    size_t IP = (size_t)W*H;

    for (int i=0;i<streams;i++){
        cudaMalloc(&dIn[i],   IP*sizeof(unsigned char));
        cudaMalloc(&dTmp[i],  IP*sizeof(float));
        cudaMalloc(&dBlur[i], IP*sizeof(unsigned char));
        cudaMalloc(&dEdge[i], IP*sizeof(unsigned char));
    }

    dim3 block(16,16);
    dim3 grid( (W+block.x-1)/block.x, (H+block.y-1)/block.y );

    CudaTimer tAll; tAll.start();
    float ms_h2d=0, ms_blur=0, ms_sobel=0, ms_d2h=0;
    CudaTimer th2d, tblur, tsobel, td2h;

    // Procesa N imágenes en lotes “streamed”
    int i=0;
    while (i < N){
        int batch = std::min(streams, N - i);

        // Genera batch sintético y lanza pipeline en streams
        std::vector<std::vector<unsigned char>> hostBatch(batch);
        for (int b=0;b<batch;b++) synth_image(hostBatch[b], W, H, i+b);

        // H2D
        th2d.start();
        for (int b=0;b<batch;b++){
            cudaMemcpyAsync(dIn[b], hostBatch[b].data(), IP, cudaMemcpyHostToDevice, S[b]);
        }
        ms_h2d += th2d.stop();

        // Blur separable
        tblur.start();
        for (int b=0;b<batch;b++){
            gauss1d_h<<<grid, block, 0, S[b]>>>(dIn[b], dTmp[b], W, H, dK, R);
            gauss1d_v<<<grid, block, 0, S[b]>>>(dTmp[b], dBlur[b], W, H, dK, R);
        }
        ms_blur += tblur.stop();

        // Sobel
        tsobel.start();
        for (int b=0;b<batch;b++){
            sobel_mag<<<grid, block, 0, S[b]>>>(dBlur[b], dEdge[b], W, H);
        }
        ms_sobel += tsobel.stop();

        // D2H y guardar
        td2h.start();
        for (int b=0;b<batch;b++){
            std::vector<unsigned char> outEdge(IP);
            cudaMemcpyAsync(outEdge.data(), dEdge[b], IP, cudaMemcpyDeviceToHost, S[b]);
            cudaStreamSynchronize(S[b]); // sincroniza por stream para guardar

            char path[256];
            snprintf(path, sizeof(path), "out/results/img_%05d_edges.pgm", i+b);
            write_pgm(path, outEdge.data(), W, H);

            // también guarda blur (opcional)
            std::vector<unsigned char> outBlur(IP);
            cudaMemcpy(outBlur.data(), dBlur[b], IP, cudaMemcpyDeviceToHost);
            snprintf(path, sizeof(path), "out/results/img_%05d_blur.pgm", i+b);
            write_pgm(path, outBlur.data(), W, H);
        }
        ms_d2h += td2h.stop();

        i += batch;
    }

    float total_ms = tAll.stop();

    {
    ensure_dir("out");
    ensure_dir("out/logs");
    std::ofstream csv("out/logs/timings.csv", std::ios::app);
    // header si quieres, solo la 1ª vez (simple, sin check de existencia)
    static bool header_printed = false;
    if (!header_printed) { 
        csv << "W,H,N,streams,sigma,H2D_ms,BLUR_ms,SOBEL_ms,D2H_ms,TOTAL_ms\n"; 
        header_printed = true;
    }
    csv << W << "," << H << "," << N << "," << streams << "," << sigma << ","
        << ms_h2d << "," << ms_blur << "," << ms_sobel << "," << ms_d2h << "," << total_ms << "\n";
}


    printf("Timing(ms): H2D=%.3f  BLUR=%.3f  SOBEL=%.3f  D2H=%.3f  TOTAL=%.3f\n",
           ms_h2d, ms_blur, ms_sobel, ms_d2h, total_ms);

    // Limpieza
    for (int s=0;s<streams;s++){
        cudaFree(dIn[s]); cudaFree(dTmp[s]); cudaFree(dBlur[s]); cudaFree(dEdge[s]);
        cudaStreamDestroy(S[s]);
    }
    cudaFree(dK);
    cudaDeviceSynchronize();
    return 0;
}
