A blog/book series taking readers from GPU fundamentals to running LLMs across multi-node clusters. Focus: NVIDIA GPUs, CUDA programming, practical expertise. 

| #   | Title                            | Focus                                                                                |
| --- | ----------------------------- |------------------------------------------------------------------———- |
| 1   | Inside the GPU              | NVIDIA architecture deep dive; SM, warp, memory hierarchy; brief AMD/TPU comparison |
| 2   | Your First CUDA Kernels  | Thread model, vec add, matmul from scratch                  |
| 3   | GPU Memory: The Real Bottleneck  | HBM, shared memory, coalescing, bank conflicts |
| 4   | CUDA Streams & Async Execution   | Concurrency, pipelining, overlapping transfers|
| 5   | Building a Neural Net on One GPU | Dense layers, activations, forward + backward |
| 6   | Attention on One GPU            | Self-attention kernel, FlashAttention concepts         |
| 7   | Mixed Precision & Quantization   | FP16/BF16/FP8, loss scaling, practical tradeoffs  |
| 8   | Multi-GPU Infrastructure       | NVLink, IB, RDMA, NCCL collectives                       |
| 9   | Data Parallel & Tensor Parallel  | AllReduce, sharding strategies, a working example|
| 10  | Pipeline Parallel & MoE         | Micro-batching, expert routing                                |
| 11  | Profiling & Optimization        | Nsight, roofline model, identifying bottlenecks         |
| 12  | Inference on one GPU           | Prefill, Decode, and the KV Cache                            |
| 13  | Distributed Inference            | Sharding Models Across GPUs                                    |
| 14  | Debugging & Reliability        | cuda-gdb, error handling, silent corruption             |
