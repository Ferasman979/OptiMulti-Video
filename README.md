# OptiMulti-Video

**High-Performance Multimodal Attention with Custom CUDA Kernels**

This project demonstrates a vertical slice of a high-performance Multimodal AI system. It features:
1.  **Custom CUDA Kernel**: A fused "Normalize & Project" kernel written in CUDA C++ for low-latency fusion of video and text embeddings.
2.  **Multimodal Architecture**: A compact Video-Text Transformer model.
3.  **Distributed Training**: FSDP (Fully Sharded Data Parallel) training loop designed to run on dual T4 GPUs (available for free on Kaggle/Colab).

##  How to Run on Google Colab / Kaggle For Testing Purposes:

1.  **Push to GitHub**: Sync this local folder to a public GitHub repository named `OptiMulti-Video`.
2.  **Open the Notebook**: Upload `notebooks/colab_demo.ipynb` to Google Colab.
3.  **Run**: The notebook is pre-configured to:
    *   Clone your repository.
    *   Install dependencies.
    *   Compile the Custom CUDA kernel (JIT compilation).
    *   Run the FSDP training demo.

## Project Structure

- `src/`: C++ and CUDA source code for the custom kernel.
- `model/`: PyTorch model definitions (Vision Encoder, Text Decoder, Fusion).
- `training/`: Training loops and distributed data configuration.
- `notebooks/`: Jupyter notebooks for demonstration.
