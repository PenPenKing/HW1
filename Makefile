# Makefile -- conventional CUDA build (Linux / H20 cluster, nvcc available).
#
#   make                 # picks nvcc when present, else the NVRTC back-end
#   make BACKEND=nvrtc   # force the NVRTC back-end (g++ + libnvrtc)
#   make BACKEND=nvcc    # force nvcc
#   make test            # run both implementations
#
# On Windows without MSVC use build.ps1 instead.

NVCC   ?= nvcc
CXX    ?= g++
ARCH   ?= -arch=native
OUT    ?= build
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
BACKEND ?= $(shell command -v $(NVCC) >/dev/null 2>&1 && echo nvcc || echo nvrtc)

ifeq ($(BACKEND),nvcc)
  OBJS = $(OUT)/main.o $(OUT)/launch_nvcc.o
  LIBS = -lcudart
else
  OBJS = $(OUT)/main.o $(OUT)/launch_nvrtc.o
  LIBS = -lcudart -lnvrtc
endif

all: $(OUT)/nut_lenet

$(OUT):
	mkdir -p $(OUT)

$(OUT)/main.o: main.cpp launch.h | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/launch_nvrtc.o: launch_nvrtc.cpp launch.h | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/launch_nvcc.o: launch_nvcc.cu kernels.cu launch.h | $(OUT)
	$(NVCC) $(ARCH) -O2 -std=c++17 -c $< -o $@

$(OUT)/nut_lenet: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

test: all
	./$(OUT)/nut_lenet --impl shared
	./$(OUT)/nut_lenet --impl naive --iters 5

clean:
	rm -rf $(OUT)

.PHONY: all test clean
