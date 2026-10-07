# 大作业一 · 基本题：手写 CUDA 实现 CNN 推理（MNIST）

基于 CUDA 从零实现 LeNet 风格的 CNN 推理，**不使用 cuDNN / cuBLAS / cuFFT 等任何
NVIDIA 数学库**，卷积、池化、激活、全连接、分类全部手写 kernel。
权重由本仓库附带的纯 NumPy 训练脚本产生（同样是手写前向与反向，不依赖 PyTorch）。

```
输入 1x28x28
  └─ conv 1→8, 3x3, stride 1, pad 1 ─ ReLU ─ maxpool 2x2 ─→  8x14x14
  └─ conv 8→16, 3x3, stride 1, pad 1 ─ ReLU ─ maxpool 2x2 ─→ 16x7x7
  └─ flatten(784) ─ dense 784→10 ─ argmax
共 9098 个参数
```

---

## 1. 结果摘要

| 项目 | 数值 |
|---|---|
| 测试集准确率 | **99.04 %**（9904 / 10000，MNIST 官方测试集） |
| 与 NumPy 参考实现一致性 | 10000 / 10000 张分类结果完全相同；各层最大绝对误差 ≤ 2.9e-6 |
| 设备计算时间（10000 张，输入常驻显存） | **1.318 ms** → 0.132 µs/张 → 7.59 M img/s |
| 端到端时间（含每批 host→device 拷贝） | 6.469 ms → 0.647 µs/张 |
| 硬件 | NVIDIA GeForce RTX 4090 D (sm_89), CUDA 12.9 |

> 详细的分档对比见 [§5 性能优化记录](#5-性能优化记录)，原始数据见 `results/`。

---

## 2. 文件清单

| 文件 | 说明 |
|---|---|
| `kernels.cu` | **全部 device 代码**：conv / relu / pool / dense / argmax 的朴素版与优化版。只含 `__global__` 函数，不含 host 代码，因此 nvcc 与 NVRTC 可以编译同一份源码 |
| `main.cpp` | host 编排：读 IDX、读权重、逐层发射、计时、准确率统计、逐层 dump（可移植，不含 `<<<>>>`） |
| `launch.h` | 发射层抽象：`gpu::init / gpu::kernel(name)` |
| `launch_nvrtc.cpp` | NVRTC 后端（无 nvcc 环境时用，见 §7） |
| `launch_nvcc.cu` | nvcc 后端（H20 等有完整 CUDA 工具链的机器） |
| `mnist_loader.py` | 下载 MNIST（HF 镜像）并导出 idx / npy |
| `train_numpy.py` | 纯 NumPy 训练 LeNet（手写 im2col 卷积、池化、反向传播、Adam），导出权重与参考结果 |
| `verify.py` | 把 CUDA 输出与 NumPy 参考逐层、端到端比对 |
| `build.ps1` / `run.ps1` | Windows 构建与运行 |
| `Makefile` | Linux / nvcc 构建 |
| `bench.ps1` | 一键复现性能对比表 |
| `data/` | MNIST（脚本生成，未入库） |
| `weights/` | 训练产出的权重与参考数据 |
| `results/` | 实测结果 JSON / CSV / Markdown |

---

## 3. 快速开始

### 3.1 获取数据集

```powershell
python mnist_loader.py --out data
```

脚本从 HuggingFace 的 `ylecun/mnist` 拉取 parquet（自动走 `HF_ENDPOINT`，国内可用
`hf-mirror.com`），解出图像后导出为**原始 IDX 格式**与 npy：

```
data/mnist/train-images-idx3-ubyte   60000x28x28
data/mnist/train-labels-idx1-ubyte
data/mnist/t10k-images-idx3-ubyte    10000x28x28
data/mnist/t10k-labels-idx1-ubyte
```

脚本会校验 MNIST 的已知统计量（像素均值 33.3184、各类别样本数），确保数据没下错。

### 3.2 训练权重

```powershell
python train_numpy.py --epochs 6 --out weights
```

可选 `--augment 2`（随机平移增强）、`--cosine`（余弦学习率衰减）。

训练配方与实测测试集准确率（TinyLeNet，9098 参数）：

| 配方 | 测试准确率 |
|---|---|
| 6 epochs, Adam 2e-3 | 98.68 % |
| 16 epochs, Adam 2e-3 | 98.64 %（train 已到 99.5 %，属过拟合，加轮次无益） |
| **14 epochs, Adam 3e-3, `--augment 2`, `--cosine`** | **99.04 %** ← 仓库中 `weights/` 用的就是这一版 |

产物：

* `weights/lenet_weights.bin` — 9098 个 float32，顺序为
  `conv1_w(8,1,3,3) | conv1_b(8) | conv2_w(16,8,3,3) | conv2_b(16) | fc_w(10,784) | fc_b(10)`，
  权重为 OIHW
* `weights/lenet_weights.json` — 各张量的形状与偏移
* `weights/ref_test_logits.npy` — 全测试集的 NumPy 参考 logits
* `weights/ref_layers.npz` — 前 8 张图各层中间结果，用于逐层对拍

### 3.3 构建

**Windows（无 Visual Studio）**

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

**Linux / H20（有 nvcc）**

```bash
make                # 自动检测 nvcc，没有则退回 NVRTC 后端
make test           # 跑两个实现
```

### 3.4 运行

```powershell
# 默认配置（conv1 用 shared、conv2 用 row），全测试集，20 次计时取最优
powershell -ExecutionPolicy Bypass -File run.ps1 --impl best --batch 2000 --resident

# 朴素实现（对照组）
powershell -ExecutionPolicy Bypass -File run.ps1 --impl naive

# 导出结果供校验
.\build\nut_lenet.exe --dump-layers build/dump --dump-logits build/dump/logits.bin
```

常用参数：`--impl naive|shared|row|best`、`--batch N`、`--resident`
（输入常驻显存，只计 kernel 时间）、`--iters`、`--warmup`、`--n`。

### 3.5 正确性验证

```powershell
python verify.py --dump build/dump --weights weights
```

输出（实测）：

```
layer-by-layer (CUDA vs NumPy):
  conv1    max|diff|=7.153e-07  mean|diff|=1.529e-08  OK
  relu1    max|diff|=5.178e-07  mean|diff|=7.938e-09  OK
  pool1    max|diff|=4.768e-07  mean|diff|=1.340e-08  OK
  conv2    max|diff|=2.861e-06  mean|diff|=9.592e-08  OK
  relu2    max|diff|=1.431e-06  mean|diff|=1.960e-08  OK
  pool2    max|diff|=1.431e-06  mean|diff|=4.279e-08  OK
end to end (logits):
  logits   max|diff|=3.052e-05  mean|diff|=1.928e-06  OK
  argmax   identical for 10000/10000 images OK
  accuracy CUDA 99.04%   NumPy 99.04%
RESULT: PASS
```

---

## 4. 算子实现说明

数据布局统一为 **NCHW + float32 连续**，输入归一化到 `[0,1]`（除以 255，与训练一致）。

| 算子 | kernel | 做法 |
|---|---|---|
| 卷积 | `conv_naive` | 通用版：一个线程算一个输出元素，边界用 `continue` 判断。用作正确性基准 |
| 卷积 | `conv1_shared` / `conv2_shared` | 一个 CTA 处理一张图，把整张输入（含 1 像素零 halo）搬进 shared memory，8 / 16 个输出通道复用同一份输入，权重走 `__ldg` 只读路径并在 warp 内广播 |
| 卷积 | `conv1_row` / `conv2_row` | 每个线程负责**半行 / 一整行输出**，把输入行与权重放进寄存器，shared / 权重读取被 14 个输出像素摊薄 |
| 激活 | `relu_inplace` | 原地 `fmaxf(x,0)` |
| 池化 | `maxpool2x2` | 2x2 步长 2，一个线程一个输出，四次 `fmaxf` |
| 全连接 | `dense10` | 一个线程一个 (图像, 类别)，784 次 FMA，权重与输入都走 `__ldg` |
| 分类 | `argmax10` | 一个线程一张图，10 元素全展开比较 |

**数值一致性**：卷积全部使用 `fmaf`，累加顺序与 NumPy 的 im2col 矩阵乘不同，
因此结果存在 ~1e-6 量级的差异；`--use_fast_math` 未开启，且 10000 张图的 argmax
与 NumPy **完全一致**。

---

## 5. 性能优化记录

batch = 2000，预热 5 遍，连续 20 遍取最优。`resident` = 输入已在显存（纯 kernel 时间），
`streamed` = 含每批 host→device 拷贝。单位：ms。

| 实现 | 模式 | 准确率 | 整遍 | 单张 (µs) | 吞吐 (img/s) | conv1 | conv2 | dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| naive | resident | 99.04% | 2.292 | 0.229 | 4 362 271 | 0.713 | 1.086 | 0.140 |
| shared | resident | 99.04% | 1.906 | 0.191 | 5 246 546 | 0.333 | 1.070 | 0.136 |
| row | resident | 99.04% | 2.106 | 0.211 | 4 747 508 | 1.104 | 0.488 | 0.137 |
| **best** | resident | 99.04% | **1.318** | **0.132** | **7 587 898** | 0.333 | 0.491 | 0.136 |
| naive | streamed | 99.04% | 7.461 | 0.746 | 1 340 327 | 1.033 | 1.087 | 0.137 |
| shared | streamed | 99.04% | 7.251 | 0.725 | 1 379 131 | 1.622 | 1.068 | 0.137 |
| row | streamed | 99.04% | 7.793 | 0.779 | 1 283 225 | 1.440 | 0.490 | 0.138 |
| best | streamed | 99.04% | 6.469 | 0.647 | 1 545 782 | 0.805 | 0.489 | 0.137 |

> `streamed` 模式的分层时间含每层 `cudaEventSynchronize` 的开销，只作参考；
> 整遍时间是干净的。原始数据：`results/bench_all.csv`、`results/bench_*.json`。

优化点与观察：

1. **shared memory 复用输入**：conv1 从 0.713 ms 降到 0.333 ms（2.1 倍）。朴素版每个输出
   元素都要重新从 global memory 取 9 个输入，而一张图的输入只有 784 个元素、却要产生
   6272 个输出，重复读取虽被 L1 挡住，但指令数没有减少。
2. **寄存器复用（row 版）**：conv2 从 1.086 ms 降到 0.488 ms（2.2 倍）。原来每个输出要读
   72 个权重，row 版让一行 14 个输出共用同一批权重与输入行，权重读取量降到 1/14。
3. **conv1 上 row 版反而更慢**（1.104 ms vs 0.333 ms）：整行需要 28 个累加器 + 30 个输入
   寄存器，寄存器压力大、占用率下降；即使改成 14 宽半行（448 线程）也没能追平 shared 版。
   因此最终配置 `best` = **conv1 用 shared 版、conv2 用 row 版**，各取所长。
4. **batch 大小**：2000 时最快。太小（256）时 kernel 发射开销占主导（2.8 ms），
   太大（10000）时中间张量超出 L2 容量（2.8 ms）。
5. **输入常驻显存**：把 31 MB 输入一次性拷进显存后，计时区间只剩 kernel，
   端到端 6.469 ms → 设备计算 1.318 ms，差距几乎全是 PCIe 拷贝。
6. **还能怎么优化**：conv1 的瓶颈在半精度用不上的指令吞吐（每输出 9 次 FMA 却要配套
   索引与 shared 读取），可以尝试把 8 个输出通道的权重组进向量寄存器、用 `float4`
   一次读 4 个输入；另外 dense 只有 784×10 次乘加，几乎完全被发射开销支配，
   可以把 dense+argmax 融合成一条 kernel 省一次发射。

---

## 6. 实验环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24 GB，driver 581.80 |
| 计算能力 | sm_89 |
| CUDA | 12.9（`nvrtc64_120_0` + `cudart64_12`） |
| host 编译器 | mingw-w64 g++（Windows 本机无 MSVC） |
| 数据集 | MNIST（`ylecun/mnist`，60000 训练 / 10000 测试） |

计时口径：预热 5 遍后连续跑 20 遍全测试集，取**最优值**；`cudaEvent` 插在整遍
推理前后，`--resident` 模式下不含 host↔device 拷贝。

---

## 7. 关于构建方式（为什么有两个后端）

考核平台（H20）上 CUDA 工具链完整，`make` 会直接用 `nvcc` 编译 `launch_nvcc.cu`。

但本机 Windows 上没有装 Visual Studio，而 **nvcc 在 Windows 上必须有 cl.exe**
（`nvcc -cubin` / `-ptx` 同样会去找 cl.exe，无法绕过），mingw 的 g++ 也不被
nvcc 接受。为了能在本机真实地编译、运行、验证和调优这套 kernel，额外实现了
`launch_nvrtc.cpp` 后端：

* 用 **NVRTC**（NVIDIA 官方的运行时编译库）把 `kernels.cu` 编译成 sm_89 的 PTX；
* 用 **cudart** 的 `cudaLibraryLoadData` / `cudaLibraryGetKernel` / `cudaLaunchKernelExC`
  装载并发射。

两个后端编译的是**同一份 `kernels.cu`、同一份 `main.cpp`**，只有"按名字拿 kernel
指针"这一层不同，因此本机测得的性能与 H20 上的 nvcc 构建跑的是同一套设备代码。

```bash
# H20 上强制用 nvcc
make BACKEND=nvcc
```

---

## 8. 已知限制

* 卷积 kernel 针对本网络固定尺寸（1→8 / 8→16、3x3、pad 1、28x28 / 14x14）特化，
  通用形状请用 `conv_naive`。
* 只支持 batch 推理，未实现训练反向（反向在 NumPy 侧）。
* 权重必须按 §3.2 的顺序布局；`main.cpp` 中的偏移量是编译期常量，已在文件里注明。
