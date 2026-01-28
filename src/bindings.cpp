#include <torch/extension.h>

// Forward declaration
void fused_add_layernorm_cuda(at::Tensor video, at::Tensor text, at::Tensor out, float eps);

// C++ Interface
void fused_add_layernorm(at::Tensor video, at::Tensor text, at::Tensor out, float eps) {
    // Check inputs
    TORCH_CHECK(video.type().is_cuda(), "Video tensor must be a CUDA tensor");
    TORCH_CHECK(text.type().is_cuda(), "Text tensor must be a CUDA tensor");
    TORCH_CHECK(out.type().is_cuda(), "Output tensor must be a CUDA tensor");
    TORCH_CHECK(video.sizes() == text.sizes(), "Video and Text tensors must have the same shape");
    TORCH_CHECK(video.is_contiguous(), "Video tensor must be contiguous");
    TORCH_CHECK(text.is_contiguous(), "Text tensor must be contiguous");
    
    fused_add_layernorm_cuda(video, text, out, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_layernorm", &fused_add_layernorm, "Fused Add+LayerNorm (CUDA)");
}
