# Portable Makefile for Linux/macOS reviewers
.RECIPEPREFIX := >

IDIR := ./src
COMPILER := nvcc
CXXFLAGS := -O3 --std=c++17 -I$(IDIR)
LDFLAGS := -lcudart
OUTDIR := bin
TARGET := $(OUTDIR)/image_pipeline.exe
SRCS := $(IDIR)/main.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) | $(OUTDIR)
> $(COMPILER) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(OUTDIR):
> mkdir -p $(OUTDIR)

run: $(TARGET)
> ./$(TARGET) --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6

clean:
> rm -rf $(OUTDIR)
