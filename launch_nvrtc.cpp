// launch_nvrtc.cpp -- NVRTC back-end.
//
// Compiles kernels.cu to PTX for the local GPU at start-up and loads it through
// the CUDA runtime library API (cudaLibraryLoadData / cudaLibraryGetKernel,
// CUDA 12).  This lets us build and run the real device code with nothing but a
// C++ host compiler (mingw g++ on Windows, where no MSVC toolchain exists).
#include "launch.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <nvrtc.h>

namespace gpu {
namespace {

cudaLibrary_t g_lib = nullptr;
std::unordered_map<std::string, void*> g_kernels;

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

}  // namespace

bool init(const std::string& src_path, std::string* err) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
        return fail(err, "no CUDA device visible");

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return fail(err, "cudaGetDeviceProperties failed");

    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", prop.major, prop.minor);

    std::ifstream in(src_path);
    if (!in) return fail(err, "cannot open kernel source: " + src_path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();

    nvrtcProgram prog = nullptr;
    if (nvrtcCreateProgram(&prog, src.c_str(), "kernels.cu", 0, nullptr, nullptr)
        != NVRTC_SUCCESS)
        return fail(err, "nvrtcCreateProgram failed");

    const char* opts[] = {arch, "-default-device", "--std=c++17"};
    nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);

    size_t logsz = 0;
    nvrtcGetProgramLogSize(prog, &logsz);
    if (logsz > 1) {
        std::string log(logsz, '\0');
        nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "[nvrtc] %s\n", log.c_str());
        if (cr != NVRTC_SUCCESS && err) *err = log;
    }
    if (cr != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        if (err && err->empty()) *err = std::string("nvrtc compile failed: ")
                                        + nvrtcGetErrorString(cr);
        return false;
    }

    size_t ptxsz = 0;
    nvrtcGetPTXSize(prog, &ptxsz);
    std::string ptx(ptxsz, '\0');
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    cudaError_t ce = cudaLibraryLoadData(&g_lib, ptx.data(), nullptr, nullptr, 0,
                                         nullptr, nullptr, 0);
    if (ce != cudaSuccess)
        return fail(err, std::string("cudaLibraryLoadData: ") + cudaGetErrorString(ce));

    static const char* names[] = {"conv_naive", "conv1_shared", "conv2_shared",
                                  "conv1_row", "conv2_row",
                                  "relu_inplace", "maxpool2x2", "dense10", "argmax10"};
    for (const char* n : names) {
        cudaKernel_t k = nullptr;
        if (cudaLibraryGetKernel(&k, g_lib, n) == cudaSuccess && k)
            g_kernels[n] = (void*)k;
    }
    if (g_kernels.size() != sizeof(names) / sizeof(names[0]))
        return fail(err, "some kernels were not found in the loaded library");

    // one warm-up touch so that the context is fully created
    if (cudaFree(nullptr) != cudaSuccess && cudaGetLastError() != cudaSuccess)
        return fail(err, "cudaFree(0) failed");
    return true;
}

void* kernel(const char* name) {
    auto it = g_kernels.find(name);
    return it == g_kernels.end() ? nullptr : it->second;
}

const char* backend() { return "nvrtc-runtime"; }

void shutdown() {
    if (g_lib) { cudaLibraryUnload(g_lib); g_lib = nullptr; }
    g_kernels.clear();
}

}  // namespace gpu
