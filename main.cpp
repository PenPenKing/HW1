// main.cpp -- TinyLeNet inference driver (portable host code, no nvcc needed).
//
//   load MNIST IDX + the NumPy-trained weight blob
//   run:  conv1 -> relu -> pool  -> conv2 -> relu -> pool -> dense -> argmax
//   report accuracy, per-pass wall time and a per-layer breakdown
//
// Every layer is launched through gpu::kernel(name) + cudaLaunchKernelExC, so
// the identical host code runs on top of the NVRTC back-end (Windows/mingw) and
// the nvcc back-end (Linux/H20).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "launch.h"

// --------------------------------------------------------------------------- //
// shapes of the network
// --------------------------------------------------------------------------- //
static const int IMG_H = 28, IMG_W = 28, IMG_N = IMG_H * IMG_W;   // 784
static const int C1 = 8, H1 = 28, W1 = 28;                        // conv1 out
static const int P1H = 14, P1W = 14;                              // pool1 out
static const int C2 = 16, H2 = 14, W2 = 14;                       // conv2 out
static const int P2H = 7, P2W = 7;                                // pool2 out
static const int FCIN = C2 * P2H * P2W;                           // 784
static const int FCOUT = 10;

// weight blob layout produced by train_numpy.py (offsets in floats)
//   conv1_w  72   conv1_b   8   conv2_w 1152
//   conv2_b  16   fc_w   7840   fc_b     10        total 9098
static const int OFF_C1W = 0,    N_C1W = 8 * 1 * 3 * 3;      // 72
static const int OFF_C1B = 72,   N_C1B = 8;
static const int OFF_C2W = 80,   N_C2W = 16 * 8 * 3 * 3;     // 1152
static const int OFF_C2B = 1232, N_C2B = 16;
static const int OFF_FCW = 1248, N_FCW = 10 * FCIN;          // 7840
static const int OFF_FCB = 9088, N_FCB = 10;
static const int TOTAL_FLOATS = 9098;

// --------------------------------------------------------------------------- //
// small helpers
// --------------------------------------------------------------------------- //
#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t e_ = (expr);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

static bool read_file(const std::string& path, std::vector<char>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    out->resize((size_t)n);
    in.read(out->data(), n);
    return (bool)in;
}

static uint32_t be32(const unsigned char* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

// Load an IDX file, verify its magic and return the payload as float32 in [0,1].
static std::vector<float> load_idx_images(const std::string& path, int* count) {
    std::vector<char> raw;
    if (!read_file(path, &raw)) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    const unsigned char* p = (const unsigned char*)raw.data();
    if (be32(p) != 0x00000803u) {
        std::fprintf(stderr, "%s: not an IDX image file\n", path.c_str());
        std::exit(2);
    }
    const int n = (int)be32(p + 4), rows = (int)be32(p + 8), cols = (int)be32(p + 12);
    if (rows * cols != IMG_N) {
        std::fprintf(stderr, "%s: unexpected image size %dx%d\n", path.c_str(), rows, cols);
        std::exit(2);
    }
    if (raw.size() < 16u + (size_t)n * IMG_N) {
        std::fprintf(stderr, "%s: truncated\n", path.c_str());
        std::exit(2);
    }
    std::vector<float> img((size_t)n * IMG_N);
    const unsigned char* src = p + 16;
    for (size_t i = 0; i < img.size(); ++i) img[i] = src[i] * (1.0f / 255.0f);
    *count = n;
    return img;
}

static std::vector<int> load_idx_labels(const std::string& path, int* count) {
    std::vector<char> raw;
    if (!read_file(path, &raw)) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    const unsigned char* p = (const unsigned char*)raw.data();
    if (be32(p) != 0x00000801u) {
        std::fprintf(stderr, "%s: not an IDX label file\n", path.c_str());
        std::exit(2);
    }
    const int n = (int)be32(p + 4);
    std::vector<int> lab(n);
    for (int i = 0; i < n; ++i) lab[i] = p[8 + i];
    *count = n;
    return lab;
}

static void upload(const std::vector<float>& src, float* dst) {
    CUDA_OK(cudaMemcpy(dst, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// --------------------------------------------------------------------------- //
// launch wrapper
// --------------------------------------------------------------------------- //
struct Gpu {
    float *c1w, *c1b, *c2w, *c2b, *fcw, *fcb;
    float *x, *y1, *y2, *y3, *y4, *logits;
    int* pred;
    int batch = 0;
    bool resident = false;      // inputs preloaded on the device
    float* d_all = nullptr;     // n_images x 784 resident input buffer

    void launch(const char* name, int grid, int block, void** args) {
        void* fn = gpu::kernel(name);
        if (!fn) { std::fprintf(stderr, "kernel %s not found\n", name); std::exit(2); }
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((unsigned)grid);
        cfg.blockDim = dim3((unsigned)block);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = nullptr;
        CUDA_OK(cudaLaunchKernelExC(&cfg, (const void*)fn, args));
    }
};

enum class Impl { Naive, Shared, Row, Best };

static const char* impl_name(Impl i) {
    switch (i) {
        case Impl::Naive:  return "naive";
        case Impl::Shared: return "shared";
        case Impl::Row:    return "row";
        default:           return "best (conv1 shared + conv2 row)";
    }
}

// Run the whole network once over `n` images starting at `start`.
// If `timer` is non-null the per-layer milliseconds are accumulated into it.
struct LayerTimes {
    double conv1 = 0, relu1 = 0, pool1 = 0, conv2 = 0, relu2 = 0, pool2 = 0,
           dense = 0, argmax = 0;
};

struct Ev {
    cudaEvent_t a, b;
    void create() { CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b)); }
    void tic() { CUDA_OK(cudaEventRecord(a)); }
    double toc() {
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

static void run_pass(Gpu& g, const float* host_x, const int* host_labels, int start,
                     int n, Impl impl, LayerTimes* lt, int* correct_out,
                     float* logits_out, int* pred_out, const std::string& dump_dir,
                     int dump_images) {
    const int B = g.batch;
    std::vector<int> h_pred;
    if (correct_out) h_pred.resize(B);

    LayerTimes local;
    Ev ev;
    if (lt) ev.create();

    for (int base = start; base < start + n; base += B) {
        int b = std::min(B, start + n - base);

        if (g.resident) {
            // inputs were uploaded once; point the first layer at the slice
            g.x = g.d_all + (size_t)base * IMG_N;
        } else {
            CUDA_OK(cudaMemcpy(g.x, host_x + (size_t)base * IMG_N,
                               (size_t)b * IMG_N * 4, cudaMemcpyHostToDevice));
        }

        // ---- conv1 -------------------------------------------------------
        if (lt) ev.tic();
        if (impl == Impl::Naive) {
            const int total = b * C1 * H1 * W1, CI = 1, CO = C1;
            float* y = g.y1; const float* x = g.x, *w = g.c1w, *bb = g.c1b;
            void* args[] = {&x, &w, &bb, &y, &b, (void*)&CI, (void*)&CO,
                            (void*)&H1, (void*)&W1};
            g.launch("conv_naive", (total + 255) / 256, 256, args);
        } else if (impl == Impl::Shared || impl == Impl::Best) {
            const float* x = g.x, *w = g.c1w, *bb = g.c1b;
            float* y = g.y1;
            void* args[] = {&x, &w, &bb, &y};
            g.launch("conv1_shared", b, 256, args);
        } else {
            const float* x = g.x, *w = g.c1w, *bb = g.c1b;
            float* y = g.y1;
            void* args[] = {&x, &w, &bb, &y};
            g.launch("conv1_row", b, 448, args);
        }
        if (lt) local.conv1 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C1 * H1 * W1;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y1, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/conv1.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- relu1 -------------------------------------------------------
        if (lt) ev.tic();
        {
            const int total = b * C1 * H1 * W1;
            float* y = g.y1;
            void* args[] = {&y, (void*)&total};
            g.launch("relu_inplace", (total + 255) / 256, 256, args);
        }
        if (lt) local.relu1 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C1 * H1 * W1;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y1, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/relu1.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- pool1 -------------------------------------------------------
        if (lt) ev.tic();
        {
            const int total = b * C1 * P1H * P1W;
            const float* x = g.y1; float* y = g.y2;
            int N = b, C = C1, H = H1, W = W1;
            void* args[] = {&x, &y, &N, &C, &H, &W};
            g.launch("maxpool2x2", (total + 255) / 256, 256, args);
        }
        if (lt) local.pool1 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C1 * P1H * P1W;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y2, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/pool1.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- conv2 -------------------------------------------------------
        if (lt) ev.tic();
        if (impl == Impl::Naive) {
            const int total = b * C2 * H2 * W2, CI = C1, CO = C2;
            const float* x = g.y2, *w = g.c2w, *bb = g.c2b;
            float* y = g.y3;
            void* args[] = {&x, &w, &bb, &y, &b, (void*)&CI, (void*)&CO,
                            (void*)&H2, (void*)&W2};
            g.launch("conv_naive", (total + 255) / 256, 256, args);
        } else if (impl == Impl::Shared) {
            const float* x = g.y2, *w = g.c2w, *bb = g.c2b;
            float* y = g.y3;
            void* args[] = {&x, &w, &bb, &y};
            g.launch("conv2_shared", b, 256, args);
        } else {
            const float* x = g.y2, *w = g.c2w, *bb = g.c2b;
            float* y = g.y3;
            void* args[] = {&x, &w, &bb, &y};
            g.launch("conv2_row", b, 224, args);
        }
        if (lt) local.conv2 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C2 * H2 * W2;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y3, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/conv2.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- relu2 -------------------------------------------------------
        if (lt) ev.tic();
        {
            const int total = b * C2 * H2 * W2;
            float* y = g.y3;
            void* args[] = {&y, (void*)&total};
            g.launch("relu_inplace", (total + 255) / 256, 256, args);
        }
        if (lt) local.relu2 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C2 * H2 * W2;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y3, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/relu2.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- pool2 -------------------------------------------------------
        if (lt) ev.tic();
        {
            const int total = b * C2 * P2H * P2W;
            const float* x = g.y3; float* y = g.y4;
            int N = b, C = C2, H = H2, W = W2;
            void* args[] = {&x, &y, &N, &C, &H, &W};
            g.launch("maxpool2x2", (total + 255) / 256, 256, args);
        }
        if (lt) local.pool2 += ev.toc();

        if (dump_images > 0 && dump_dir.size() && base == start) {
            const size_t sz = (size_t)std::min(dump_images, b) * C2 * P2H * P2W;
            std::vector<float> tmp(sz);
            CUDA_OK(cudaMemcpy(tmp.data(), g.y4, sz * 4, cudaMemcpyDeviceToHost));
            std::ofstream(dump_dir + "/pool2.bin", std::ios::binary)
                .write((char*)tmp.data(), sz * 4);
        }

        // ---- dense -------------------------------------------------------
        if (lt) ev.tic();
        {
            const int total = b * FCOUT;
            const float* x = g.y4, *w = g.fcw, *bb = g.fcb;
            float* y = g.logits;
            void* args[] = {&x, &w, &bb, &y, &b};
            g.launch("dense10", (total + 127) / 128, 128, args);
        }
        if (lt) local.dense += ev.toc();

        // ---- argmax ------------------------------------------------------
        if (lt) ev.tic();
        {
            const float* l = g.logits; int* p = g.pred;
            void* args[] = {&l, &p, &b};
            g.launch("argmax10", (b + 127) / 128, 128, args);
        }
        if (lt) local.argmax += ev.toc();
        // ---- collect -----------------------------------------------------
        if (correct_out) {
            CUDA_OK(cudaMemcpy(h_pred.data(), g.pred, (size_t)b * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < b; ++i)
                correct_out[0] += (h_pred[i] == host_labels[base + i]) ? 1 : 0;
            if (pred_out)
                std::copy(h_pred.begin(), h_pred.begin() + b, pred_out + (base - start));
        }
        if (logits_out)
            CUDA_OK(cudaMemcpy(logits_out + (size_t)(base - start) * FCOUT, g.logits,
                               (size_t)b * FCOUT * 4, cudaMemcpyDeviceToHost));
    }

    if (lt) {
        lt->conv1 += local.conv1;
        lt->relu1 += local.relu1;
        lt->pool1 += local.pool1;
        lt->conv2 += local.conv2;
        lt->relu2 += local.relu2;
        lt->pool2 += local.pool2;
        lt->dense += local.dense;
        lt->argmax += local.argmax;
        CUDA_OK(cudaEventDestroy(ev.a));
        CUDA_OK(cudaEventDestroy(ev.b));
    }
}

// --------------------------------------------------------------------------- //
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // keep output on crashes
    std::string data = "data/mnist", wpath = "weights/lenet_weights.bin";
    std::string kernels = "kernels.cu", impl_s = "shared", dump_dir;
    std::string out_json, logits_file;   // --out FILE to save the run summary as JSON
    int n_images = -1, batch = 256, iters = 20, warmup = 3, dump_images = 0;
    bool resident = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if (a == "--data") data = next();
        else if (a == "--weights") wpath = next();
        else if (a == "--kernels") kernels = next();
        else if (a == "--impl") impl_s = next();
        else if (a == "--n") n_images = std::atoi(next().c_str());
        else if (a == "--batch") batch = std::atoi(next().c_str());
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--dump-layers") { dump_dir = next(); dump_images = 8; }
        else if (a == "--dump-images") dump_images = std::atoi(next().c_str());
        else if (a == "--dump-logits") logits_file = next();
        else if (a == "--resident") resident = true;
        else if (a == "--out") out_json = next();
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--impl naive|shared] [--n N] [--batch B] "
                        "[--iters I] [--warmup W] [--dump-layers DIR] [--dump-logits F]\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const Impl impl = (impl_s == "naive")  ? Impl::Naive
                      : (impl_s == "row")  ? Impl::Row
                      : (impl_s == "best") ? Impl::Best
                                           : Impl::Shared;

    // ---- data -------------------------------------------------------------
    int ntest = 0, nlab = 0;
    std::vector<float> test = load_idx_images(data + "/t10k-images-idx3-ubyte", &ntest);
    std::vector<int> labels = load_idx_labels(data + "/t10k-labels-idx1-ubyte", &nlab);
    if (ntest != nlab) { std::fprintf(stderr, "image/label count mismatch\n"); return 2; }
    if (n_images < 0 || n_images > ntest) n_images = ntest;
    if (batch > n_images) batch = n_images;

    // ---- weights ----------------------------------------------------------
    std::vector<char> wraw;
    if (!read_file(wpath, &wraw) || wraw.size() != (size_t)TOTAL_FLOATS * 4) {
        std::fprintf(stderr, "%s: expected %d floats\n", wpath.c_str(), TOTAL_FLOATS);
        return 2;
    }
    const float* wf = (const float*)wraw.data();
    auto part = [&](int off, int cnt) {
        return std::vector<float>(wf + off, wf + off + cnt);
    };
    std::vector<float> c1w = part(OFF_C1W, N_C1W), c1b = part(OFF_C1B, N_C1B);
    std::vector<float> c2w = part(OFF_C2W, N_C2W), c2b = part(OFF_C2B, N_C2B);
    std::vector<float> fcw = part(OFF_FCW, N_FCW), fcb = part(OFF_FCB, N_FCB);

    // ---- device -----------------------------------------------------------
    std::string err;
    if (!gpu::init(kernels, &err)) {
        std::fprintf(stderr, "gpu::init failed: %s\n", err.c_str());
        return 2;
    }
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    Gpu g{};
    g.batch = batch;
    CUDA_OK(cudaMalloc(&g.c1w, c1w.size() * 4)); upload(c1w, g.c1w);
    CUDA_OK(cudaMalloc(&g.c1b, c1b.size() * 4)); upload(c1b, g.c1b);
    CUDA_OK(cudaMalloc(&g.c2w, c2w.size() * 4)); upload(c2w, g.c2w);
    CUDA_OK(cudaMalloc(&g.c2b, c2b.size() * 4)); upload(c2b, g.c2b);
    CUDA_OK(cudaMalloc(&g.fcw, fcw.size() * 4)); upload(fcw, g.fcw);
    CUDA_OK(cudaMalloc(&g.fcb, fcb.size() * 4)); upload(fcb, g.fcb);

    CUDA_OK(cudaMalloc(&g.x, (size_t)batch * IMG_N * 4));
    CUDA_OK(cudaMalloc(&g.y1, (size_t)batch * C1 * H1 * W1 * 4));
    CUDA_OK(cudaMalloc(&g.y2, (size_t)batch * C1 * P1H * P1W * 4));
    CUDA_OK(cudaMalloc(&g.y3, (size_t)batch * C2 * H2 * W2 * 4));
    CUDA_OK(cudaMalloc(&g.y4, (size_t)batch * C2 * P2H * P2W * 4));
    CUDA_OK(cudaMalloc(&g.logits, (size_t)batch * FCOUT * 4));
    CUDA_OK(cudaMalloc(&g.pred, (size_t)batch * FCOUT * 4));

    if (resident) {          // keep the whole test set on the device so that the
        g.resident = true;   // timed loop measures kernels only, not PCIe copies
        CUDA_OK(cudaMalloc(&g.d_all, (size_t)n_images * IMG_N * 4));
        CUDA_OK(cudaMemcpy(g.d_all, test.data(), (size_t)n_images * IMG_N * 4,
                           cudaMemcpyHostToDevice));
    }

    // ---- dump + accuracy ---------------------------------------------------
    std::vector<float> all_logits;
    if (!logits_file.empty()) all_logits.resize((size_t)n_images * FCOUT);
    int correct = 0;
    run_pass(g, test.data(), labels.data(), 0, n_images, impl, nullptr, &correct,
             all_logits.empty() ? nullptr : all_logits.data(),
             nullptr, dump_dir, dump_images);
    const double acc = 100.0 * correct / n_images;

    if (!logits_file.empty()) {
        std::ofstream(logits_file, std::ios::binary)
            .write((char*)all_logits.data(), all_logits.size() * 4);
        std::printf("wrote %s\n", logits_file.c_str());
    }
    if (dump_images > 0 && !dump_dir.empty())
        std::printf("dumped first %d images' layer outputs to %s\n",
                    dump_images, dump_dir.c_str());

    // ---- timing ------------------------------------------------------------
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i)
        run_pass(g, test.data(), labels.data(), 0, n_images, impl, nullptr, nullptr,
                 nullptr, nullptr, std::string(), 0);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<double> pass_ms;
    for (int i = 0; i < iters; ++i) {
        CUDA_OK(cudaEventRecord(e0));
        run_pass(g, test.data(), labels.data(), 0, n_images, impl, nullptr, nullptr,
                 nullptr, nullptr, std::string(), 0);
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        pass_ms.push_back(ms);
    }
    std::sort(pass_ms.begin(), pass_ms.end());
    const double best = pass_ms.front(), mean = [&] {
        double s = 0; for (double v : pass_ms) s += v; return s / pass_ms.size();
    }();

    // per-layer breakdown over one pass
    LayerTimes lt;
    run_pass(g, test.data(), labels.data(), 0, n_images, impl, &lt, nullptr, nullptr,
             nullptr, std::string(), 0);

    // ---- report ------------------------------------------------------------
    std::printf("\n");
    std::printf("backend        %s\n", gpu::backend());
    std::printf("device         %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("impl           %s\n", impl_name(impl));
    std::printf("images         %d   batch %d   resident %s\n", n_images, batch,
                resident ? "yes (kernels only)" : "no (includes H2D copies)");
    std::printf("accuracy       %.2f%%  (%d/%d)\n", acc, correct, n_images);
    std::printf("pass  best     %.3f ms   mean %.3f ms  (iters=%d warmup=%d)\n",
                best, mean, iters, warmup);
    std::printf("per image      %.4f ms   throughput %.1f img/s\n",
                best / n_images, n_images / (best / 1000.0));
    std::printf("layers (ms)    conv1 %.3f  relu1 %.3f  pool1 %.3f  conv2 %.3f  "
                "relu2 %.3f  pool2 %.3f  dense %.3f  argmax %.3f\n",
                lt.conv1, lt.relu1, lt.pool1, lt.conv2, lt.relu2, lt.pool2,
                lt.dense, lt.argmax);
    const double conv_ms = lt.conv1 + lt.conv2;
    std::printf("conv share     %.1f%% of the pass\n", 100.0 * conv_ms / best);

    // ---- json --------------------------------------------------------------
    if (!out_json.empty()) {
        std::FILE* f = std::fopen(out_json.c_str(), "w");
        if (f) {
            std::fprintf(f,
                "{\n  \"backend\": \"%s\",\n  \"device\": \"%s\",\n  \"impl\": \"%s\",\n"
                "  \"images\": %d,\n  \"batch\": %d,\n  \"accuracy_percent\": %.4f,\n"
                "  \"pass_best_ms\": %.4f,\n  \"pass_mean_ms\": %.4f,\n"
                "  \"ms_per_image\": %.6f,\n  \"images_per_second\": %.1f,\n"
                "  \"layers_ms\": {\"conv1\": %.4f, \"relu1\": %.4f, \"pool1\": %.4f, "
                "\"conv2\": %.4f, \"relu2\": %.4f, \"pool2\": %.4f, \"dense\": %.4f, "
                "\"argmax\": %.4f},\n  \"iters\": %d,\n  \"warmup\": %d\n}\n",
                gpu::backend(), prop.name, impl_name(impl),
                n_images, batch, acc, best, mean, best / n_images,
                n_images / (best / 1000.0), lt.conv1, lt.relu1, lt.pool1, lt.conv2,
                lt.relu2, lt.pool2, lt.dense, lt.argmax, iters, warmup);
            std::fclose(f);
            std::printf("wrote %s\n", out_json.c_str());
        }
    }

    gpu::shutdown();
    return 0;
}
