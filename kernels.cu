// kernels.cu -- device code for TinyLeNet inference.
//
// This file contains ONLY device functions (no <<<>>> launch syntax, no host
// code), so that the exact same source can be compiled by
//   * nvcc               (see launch_nvcc.cu, the conventional build), and
//   * NVRTC at runtime   (see launch_nvrtc.cpp, used when no host compiler for
//                         CUDA is available on the development machine).
//
// Layouts everywhere are NCHW with contiguous float32:
//   conv1: x (N,1,28,28)  w (8,1,3,3)   OIHW   b (8)   -> y (N,8,28,28)
//   pool1: x (N,8,28,28)                                -> y (N,8,14,14)
//   conv2: x (N,8,14,14)  w (16,8,3,3)  OIHW   b (16)  -> y (N,16,14,14)
//   pool2: x (N,16,14,14)                               -> y (N,16,7,7)
//   dense: x (N,784)      w (10,784)    b (10)          -> y (N,10)
//
// All convolutions are 3x3, stride 1, zero padding 1 (output size == input size).

#define CONV1_HW 28
#define CONV1_HP 30   // H + 2*pad
#define CONV2_HW 14
#define CONV2_HP 16
#define CI2 8
#define CO2 16
#define FC_IN 784
#define FC_OUT 10

// --------------------------------------------------------------------------
// Generic reference kernel: one thread per output element, direct global reads.
// Works for any CI/CO/H/W with a 3x3 kernel and pad 1.  Used as the correctness
// baseline and as the "before optimisation" measurement.
// --------------------------------------------------------------------------
extern "C" __global__ void conv_naive(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      const float* __restrict__ b,
                                      float* __restrict__ y,
                                      int N, int CI, int CO, int H, int W) {
    const int total = N * CO * H * W;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    const int ox = t % W;
    const int oy = (t / W) % H;
    const int co = (t / (W * H)) % CO;
    const int n = t / (CO * W * H);

    const float* xn = x + (size_t)n * CI * H * W;
    const float* wc = w + (size_t)co * CI * 9;

    float sum = b[co];
    for (int ci = 0; ci < CI; ++ci) {
        const float* xc = xn + (size_t)ci * H * W;
        const float* wk = wc + ci * 9;
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            const int iy = oy + ky - 1;
            if (iy < 0 || iy >= H) continue;
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int ix = ox + kx - 1;
                if (ix < 0 || ix >= W) continue;
                sum = fmaf(__ldg(xc + iy * W + ix), __ldg(wk + ky * 3 + kx), sum);
            }
        }
    }
    y[t] = sum;
}

// --------------------------------------------------------------------------
// Optimised conv1: 1 -> 8 channels, 28x28, 3x3, pad 1.
// One CTA per image.  The whole 28x28 input is staged in shared memory once and
// reused by all 8 output channels (9x fewer global loads); weights are read
// through the read-only path and broadcast to the warp.
// --------------------------------------------------------------------------
extern "C" __global__ void conv1_shared(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b,
                                        float* __restrict__ y) {
    __shared__ float s[CONV1_HP * CONV1_HP];

    const float* xn = x + (size_t)blockIdx.x * CONV1_HW * CONV1_HW;

    // stage the input with a 1-pixel zero halo
    for (int i = threadIdx.x; i < CONV1_HP * CONV1_HP; i += blockDim.x) {
        const int ix = i % CONV1_HP;
        const int iy = i / CONV1_HP;
        const int sx = ix - 1;
        const int sy = iy - 1;
        s[i] = (sx >= 0 && sx < CONV1_HW && sy >= 0 && sy < CONV1_HW)
                   ? xn[sy * CONV1_HW + sx] : 0.0f;
    }
    __syncthreads();

    float* yn = y + (size_t)blockIdx.x * 8 * CONV1_HW * CONV1_HW;
    const int total = 8 * CONV1_HW * CONV1_HW;
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int ox = t % CONV1_HW;
        const int oy = (t / CONV1_HW) % CONV1_HW;
        const int co = t / (CONV1_HW * CONV1_HW);

        const float* wk = w + co * 9;
        const float* sp = s + oy * CONV1_HP + ox;
        float sum = b[co];
#pragma unroll
        for (int ky = 0; ky < 3; ++ky)
#pragma unroll
            for (int kx = 0; kx < 3; ++kx)
                sum = fmaf(sp[ky * CONV1_HP + kx], __ldg(wk + ky * 3 + kx), sum);
        yn[t] = sum;
    }
}

// --------------------------------------------------------------------------
// Optimised conv2: 8 -> 16 channels, 14x14, 3x3, pad 1.
// One CTA per image, all 8 input channels staged in shared memory.
// --------------------------------------------------------------------------
extern "C" __global__ void conv2_shared(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b,
                                        float* __restrict__ y) {
    __shared__ float s[CI2 * CONV2_HP * CONV2_HP];   // 8*16*16 = 8 KiB

    const float* xn = x + (size_t)blockIdx.x * CI2 * CONV2_HW * CONV2_HW;

    for (int i = threadIdx.x; i < CI2 * CONV2_HP * CONV2_HP; i += blockDim.x) {
        const int ix = i % CONV2_HP;
        const int rem = i / CONV2_HP;
        const int iy = rem % CONV2_HP;
        const int ci = rem / CONV2_HP;
        const int sx = ix - 1;
        const int sy = iy - 1;
        s[i] = (sx >= 0 && sx < CONV2_HW && sy >= 0 && sy < CONV2_HW)
                   ? xn[(ci * CONV2_HW + sy) * CONV2_HW + sx] : 0.0f;
    }
    __syncthreads();

    float* yn = y + (size_t)blockIdx.x * CO2 * CONV2_HW * CONV2_HW;
    const int total = CO2 * CONV2_HW * CONV2_HW;      // 3136
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int ox = t % CONV2_HW;
        const int oy = (t / CONV2_HW) % CONV2_HW;
        const int co = t / (CONV2_HW * CONV2_HW);

        float sum = b[co];
#pragma unroll
        for (int ci = 0; ci < CI2; ++ci) {
            const float* sp = s + ci * CONV2_HP * CONV2_HP + oy * CONV2_HP + ox;
            const float* wk = w + (co * CI2 + ci) * 9;
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx)
                    sum = fmaf(sp[ky * CONV2_HP + kx], __ldg(wk + ky * 3 + kx), sum);
        }
        yn[t] = sum;
    }
}

// --------------------------------------------------------------------------
// Row kernels: each thread produces one whole output row (fixed co, oy) and
// keeps the input row and the weights in registers, so every shared-memory and
// weight load is amortised over 14 (conv2) / 28 (conv1) output pixels.
// This is the "after optimisation" configuration.
// conv1: 448 threads (8 co x 28 oy x 2 half-rows); conv2: 224 threads (16 co x 14 oy).
// --------------------------------------------------------------------------
extern "C" __global__ void conv1_row(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     const float* __restrict__ b,
                                     float* __restrict__ y) {
    __shared__ float s[CONV1_HP * CONV1_HP];

    const float* xn = x + (size_t)blockIdx.x * CONV1_HW * CONV1_HW;
    for (int i = threadIdx.x; i < CONV1_HP * CONV1_HP; i += blockDim.x) {
        const int ix = i % CONV1_HP;
        const int iy = i / CONV1_HP;
        const int sx = ix - 1;
        const int sy = iy - 1;
        s[i] = (sx >= 0 && sx < CONV1_HW && sy >= 0 && sy < CONV1_HW)
                   ? xn[sy * CONV1_HW + sx] : 0.0f;
    }
    __syncthreads();

    const int HALF = 14;                       // outputs per thread
    const int tid = threadIdx.x;
    if (tid >= 8 * CONV1_HW * (CONV1_HW / HALF)) return;   // 448 useful threads
    const int half = tid % (CONV1_HW / HALF);
    const int oy = (tid / (CONV1_HW / HALF)) % CONV1_HW;
    const int co = tid / (CONV1_HW * (CONV1_HW / HALF));
    const int ox0 = half * HALF;

    float acc[HALF];
#pragma unroll
    for (int i = 0; i < HALF; ++i) acc[i] = b[co];

    const float* sp = s + oy * CONV1_HP + ox0;
    const float* wk = w + co * 9;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        float xr[HALF + 2];
#pragma unroll
        for (int j = 0; j < HALF + 2; ++j) xr[j] = sp[ky * CONV1_HP + j];
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const float wv = __ldg(wk + ky * 3 + kx);
#pragma unroll
            for (int ox = 0; ox < HALF; ++ox)
                acc[ox] = fmaf(xr[ox + kx], wv, acc[ox]);
        }
    }

    float* yn = y + (size_t)blockIdx.x * 8 * CONV1_HW * CONV1_HW
                + co * (CONV1_HW * CONV1_HW) + oy * CONV1_HW + ox0;
#pragma unroll
    for (int ox = 0; ox < HALF; ++ox) yn[ox] = acc[ox];
}

extern "C" __global__ void conv2_row(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     const float* __restrict__ b,
                                     float* __restrict__ y) {
    __shared__ float s[CI2 * CONV2_HP * CONV2_HP];

    const float* xn = x + (size_t)blockIdx.x * CI2 * CONV2_HW * CONV2_HW;
    for (int i = threadIdx.x; i < CI2 * CONV2_HP * CONV2_HP; i += blockDim.x) {
        const int ix = i % CONV2_HP;
        const int rem = i / CONV2_HP;
        const int iy = rem % CONV2_HP;
        const int ci = rem / CONV2_HP;
        const int sx = ix - 1;
        const int sy = iy - 1;
        s[i] = (sx >= 0 && sx < CONV2_HW && sy >= 0 && sy < CONV2_HW)
                   ? xn[(ci * CONV2_HW + sy) * CONV2_HW + sx] : 0.0f;
    }
    __syncthreads();

    const int tid = threadIdx.x;
    if (tid >= CO2 * CONV2_HW) return;            // 224 useful threads
    const int co = tid / CONV2_HW;
    const int oy = tid % CONV2_HW;

    float acc[CONV2_HW];
#pragma unroll
    for (int i = 0; i < CONV2_HW; ++i) acc[i] = b[co];

#pragma unroll
    for (int ci = 0; ci < CI2; ++ci) {
        const float* sp = s + ci * (CONV2_HP * CONV2_HP) + oy * CONV2_HP;
        const float* wc = w + (co * CI2 + ci) * 9;
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            float xr[CONV2_HP];
#pragma unroll
            for (int j = 0; j < CONV2_HP; ++j) xr[j] = sp[ky * CONV2_HP + j];
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const float wv = __ldg(wc + ky * 3 + kx);
#pragma unroll
                for (int ox = 0; ox < CONV2_HW; ++ox)
                    acc[ox] = fmaf(xr[ox + kx], wv, acc[ox]);
            }
        }
    }

    float* yn = y + (size_t)blockIdx.x * CO2 * CONV2_HW * CONV2_HW
                + co * (CONV2_HW * CONV2_HW) + oy * CONV2_HW;
#pragma unroll
    for (int ox = 0; ox < CONV2_HW; ++ox) yn[ox] = acc[ox];
}

// --------------------------------------------------------------------------
// Element-wise ReLU, in place.
// --------------------------------------------------------------------------
extern "C" __global__ void relu_inplace(float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(y[i], 0.0f);
}

// --------------------------------------------------------------------------
// 2x2 max pooling with stride 2 (H and W must be even).
// --------------------------------------------------------------------------
extern "C" __global__ void maxpool2x2(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int N, int C, int H, int W) {
    const int QH = H >> 1, QW = W >> 1;
    const int total = N * C * QH * QW;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    const int ox = t % QW;
    const int oy = (t / QW) % QH;
    const int c = (t / (QW * QH)) % C;
    const int n = t / (C * QW * QH);

    const float* xp = x + ((size_t)(n * C + c) * H + oy * 2) * W + ox * 2;
    float m = xp[0];
    m = fmaxf(m, xp[1]);
    m = fmaxf(m, xp[W]);
    m = fmaxf(m, xp[W + 1]);
    y[t] = m;
}

// --------------------------------------------------------------------------
// Fully connected 784 -> 10, one thread per (image, class).
// --------------------------------------------------------------------------
extern "C" __global__ void dense10(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ y, int N) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * FC_OUT) return;

    const int n = t / FC_OUT;
    const int co = t % FC_OUT;
    const float* xn = x + (size_t)n * FC_IN;
    const float* wc = w + co * FC_IN;

    float sum = b[co];
    for (int i = 0; i < FC_IN; ++i) sum = fmaf(__ldg(xn + i), __ldg(wc + i), sum);
    y[t] = sum;
}

// --------------------------------------------------------------------------
// Argmax over 10 logits, one thread per image.
// --------------------------------------------------------------------------
extern "C" __global__ void argmax10(const float* __restrict__ logits,
                                    int* __restrict__ pred, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* l = logits + (size_t)n * FC_OUT;
    float best = l[0];
    int bi = 0;
#pragma unroll
    for (int c = 1; c < FC_OUT; ++c)
        if (l[c] > best) { best = l[c]; bi = c; }
    pred[n] = bi;
}
