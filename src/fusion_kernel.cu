#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for Fused Add + LayerNorm
// Operation: Out = LayerNorm(Video + Text)
// Optimizations:
// 1. Fuses the element-wise add and the reduction for mean/var into one pass (or close to it).
// 2. Uses Shared Memory for reduction.

template <typename T>
__global__ void fused_add_layernorm_kernel(
    const T* __restrict__ video,
    const T* __restrict__ text,
    T* __restrict__ out,
    int N, // Batch * SeqLen (Number of "rows")
    int D, // Hidden Dimension (Number of "cols")
    float eps) {

    // One block per row (simplest strategy for D <= 1024)
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    // Pointer arithmetic for current row
    const T* row_video = video + row_idx * D;
    const T* row_text = text + row_idx * D;
    T* row_out = out + row_idx * D;

    // Shared memory for reduction
    // We need strict alignment, using float for stats
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 1. Load Data, Compute Sum and SumSq
    // We iterate over the row if D > bdim
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    for (int i = tid; i < D; i += bdim) {
        float v = static_cast<float>(row_video[i]);
        float t = static_cast<float>(row_text[i]);
        float val = v + t; // The "Fusion" Add
        thread_sum += val;
        thread_sq_sum += val * val;
        
        // Store intermediate fused value? 
        // We can't store all D in shared mem if D is large.
        // We will re-compute (ADD) or store in distinct global mem?
        // To be truly fused and "one pass" for stats, we compute stats first.
        // But to avoid reading global twice, we ideally write to registers or shared.
        // For D=768 and float32, that's 3KB. Fits in Shared.
        // Let's assume D <= 1024 (4KB). 
        if (D <= 1024) {
            s_data[i] = val; 
        }
    }

    // Reduction for Mean
    // Typical parallel reduction in shared memory
    // For simplicity/robustness in this demo, strict block reduction
    // Using warp shuffles is faster but more verbose.
    // Let's stick to basic block reduction for readability/demo.
    
    // First, reduce thread-local sums to shared memory for final reduction
    // Actually, let's use warp shuffle for the thread_sum accumulation.
    
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        thread_sq_sum += __shfl_down_sync(mask, thread_sq_sum, offset);
    }

    // First thread of each warp has the warp sum.
    // We need to collect these.
    // Allocate space for warp sums at end of s_data, assuming s_data is huge? 
    // No, let's just use raw atomicAdd for simplicity if simplicity is key, 
    // BUT atomicAdd on float is slowish? 
    // Better: Standard shared mem reduction.
    
    // Let's implement a simpler 2-pass approach if we can't fit data in SMEM.
    // But since "Performance" is key, let's assume D fits in SMEM (common for Transformer heads).
    // If D > 1024, this kernel might overflow SMEM if we cache inputs.
    // For this demo, let's assume D <= 2048 and we have dynamic SM.
    
    // Start simple: Global Read 2x is bad. 
    // We cached "val" in s_data[i] above.
    
    // Block-wide reduction using SMEM (s_stats)
    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float s_reduce_sum[32]; // For max 1024 threads (32 warps)
    __shared__ float s_reduce_sq[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_reduce_sum[warp_id] = thread_sum;
        s_reduce_sq[warp_id] = thread_sq_sum;
    }
    __syncthreads();

    // Warp 0 reduces the partial sums
    if (warp_id == 0) {
        float val_sum = (tid < (bdim / 32 + (bdim % 32 != 0))) ? s_reduce_sum[tid] : 0.0f;
        float val_sq = (tid < (bdim / 32 + (bdim % 32 != 0))) ? s_reduce_sq[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
             val_sum += __shfl_down_sync(mask, val_sum, offset);
             val_sq += __shfl_down_sync(mask, val_sq, offset);
        }
        if (tid == 0) {
             s_mean = val_sum / D;
             s_var = (val_sq / D) - (s_mean * s_mean);
        }
    }
    __syncthreads();

    // 2. Normalize and Write
    float inv_std = rsqrtf(s_var + eps);
    float mean = s_mean;

    for (int i = tid; i < D; i += bdim) {
        // Retrieve cached value
        float val = s_data[i]; 
        // Normalize
        float norm = (val - mean) * inv_std;
        // Write
        row_out[i] = static_cast<T>(norm);
    }
}

// C++ Dispatcher
void fused_add_layernorm_cuda(at::Tensor video, at::Tensor text, at::Tensor out, float eps) {
    int N = video.size(0) * video.size(1); // Batch * Seq
    int D = video.size(2); // Hidden Dim

    const int threads = 256;
    const int blocks = N;
    // Shared mem: D * sizeof(float)
    // We assume float32 internal precision
    int smem_size = D * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(video.scalar_type(), "fused_add_layernorm_kernel", ([&] {
        fused_add_layernorm_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            video.data_ptr<scalar_t>(),
            text.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, D, eps);
    }));
}
