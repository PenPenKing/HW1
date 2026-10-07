// launch.h -- thin abstraction over "give me a kernel by name".
//
// Two back-ends implement it:
//   launch_nvrtc.cpp  compiles kernels.cu with NVRTC at run time (no nvcc / no
//                     MSVC needed; used on the Windows development machine)
//   launch_nvcc.cu    includes kernels.cu and returns the __global__ stubs
//                     (the conventional build, used on the H20 Linux cluster)
#pragma once

#include <string>

namespace gpu {

// Initialise the device and make all kernels of kernels.cu available.
// src_path is the path to kernels.cu; the NVRTC back-end needs it, the nvcc
// back-end ignores it.  Returns false and fills *err on failure.
bool init(const std::string& src_path, std::string* err);

// Look up a kernel by its extern "C" name.  Returns nullptr if unknown.
void* kernel(const char* name);

// Human readable description of the back-end, for the run log.
const char* backend();

void shutdown();

}  // namespace gpu
