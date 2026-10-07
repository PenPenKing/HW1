// launch_nvcc.cu -- conventional back-end, built by nvcc (used on the H20
// cluster where a full CUDA toolchain exists).  It includes the same kernels.cu
// that the NVRTC back-end compiles at run time, so both builds run identical
// device code.
#include "launch.h"

#include <cstring>
#include <cuda_runtime.h>

#include "kernels.cu"

namespace gpu {
namespace {

struct Entry { const char* name; const void* fn; };

const Entry kTable[] = {
    {"conv_naive",   (const void*)conv_naive},
    {"conv1_shared", (const void*)conv1_shared},
    {"conv2_shared", (const void*)conv2_shared},
    {"conv1_row",    (const void*)conv1_row},
    {"conv2_row",    (const void*)conv2_row},
    {"relu_inplace", (const void*)relu_inplace},
    {"maxpool2x2",   (const void*)maxpool2x2},
    {"dense10",      (const void*)dense10},
    {"argmax10",     (const void*)argmax10},
};

}  // namespace

bool init(const std::string& /*src_path*/, std::string* err) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        if (err) *err = "no CUDA device visible";
        return false;
    }
    cudaFree(nullptr);
    return true;
}

void* kernel(const char* name) {
    for (const Entry& e : kTable)
        if (std::strcmp(e.name, name) == 0) return const_cast<void*>(e.fn);
    return nullptr;
}

const char* backend() { return "nvcc-runtime"; }

void shutdown() {}

}  // namespace gpu
