# OptiMulti-Video Walkthrough

I have successfully generated the "OptiMulti-Video" project. This codebase implements a high-performance multimodal transformer with a custom CUDA kernel, designed for distributed training on T4 GPUs (Colab/Kaggle).

## 📂 Project Overview

| Component | File Path | Description |
|-----------|-----------|-------------|
| **CUDA Kernel** | `src/fusion_kernel.cu` | Fused Add+LayerNorm kernel optimized with Shared Memory reductions. |
| **C++ Bindings** | `src/bindings.cpp` | PyTorch C++ extension bindings. |
| **Model** | `model/multimodal.py` | `OptiMultiVideo` class that uses the custom kernel via `torch.autograd.Function`. |
| **Training** | `training/train_fsdp.py` | FSDP training script to scale across multiple GPUs. |
| **Colab Demo** | `notebooks/colab_demo.ipynb` | Jupyter notebook ready to upload to Google Colab. |

## 🚀 How to Run on Google Colab

The easiest way to demonstrate this project is to use the generated notebook.

1.  **Sync to GitHub**: 
    - Initialize a git repo in `c:\Users\Feras\Desktop\Personal\Portfolio Projects\OptiMulti-Video`
    - Push it to GitHub.
2.  **Upload Notebook**:
    - Go to [colab.research.google.com](https://colab.research.google.com).
    - Upload `notebooks/colab_demo.ipynb`.
3.  **Configure**:
    - In the notebook, uncomment the `!git clone` cell and replace `YOUR_USERNAME` with your GitHub handle.
4.  **Run All**:
    - Example Output:
      ```
      [OptiMulti] CUDA Extension Loaded Successfully.
      Spawning 2 processes for FSDP training...
      [Rank 0] Addr: FullyShardedDataParallel(...)
      Step 0, Loss: 8.4321
      ...
      Training Finished. Avg Step Time: 0.1234 s
      Throughput: 259.32 samples/s
      ```

## 🛠️ Technical Details

### Custom CUDA Kernel
The kernel in `src/fusion_kernel.cu` performs a **Fused Add + LayerNorm**.
- **Standard approach**: Read A, Read B -> Write (A+B) to Global Memory -> Read (A+B) -> Compute Mean/Var -> Write Output. (3 Global Reads, 2 Global Writes).
- **Our Kernel**: Read A, Read B -> Compute Stats in Shared Memory -> Write Output. (2 Global Reads, 1 Global Write).
- **Impact**: Reduces memory bandwidth usage, which is the bottleneck for this operation.

### Distributed Training (FSDP)
The `training/train_fsdp.py` script uses `FullyShardedDataParallel`. 
- Unlike `DataParallel` (DDP), FSDP shards model parameters, gradients, and optimizer states, allowing you to fit larger models on the limited memory of T4 GPUs (16GB).
