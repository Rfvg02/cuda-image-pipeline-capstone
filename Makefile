IDIR=./src
COMPILER=nvcc
CXXFLAGS=-O3 --std=c++17 -I\
LDFLAGS=-lcudart
OUTDIR=bin
TARGET=\/image_pipeline.exe
SRCS=\/main.cu

.PHONY: all clean run

all: \

\: \ | \
\ \ \ -o \ \

\:
mkdir -p \

run: \
./\ --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6

clean:
rm -rf \
