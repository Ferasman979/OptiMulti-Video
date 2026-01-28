import torch
import torch.nn as nn
from torch.autograd import Function

# Try to import the compiled extension
try:
    import optimulti_fusion_cuda
    print("[OptiMulti] CUDA Extension Loaded Successfully.")
except ImportError:
    print("[OptiMulti] Warning: CUDA Extension not found. Run 'python setup.py install'. Fallback to PyTorch implementation.")
    optimulti_fusion_cuda = None

class FusedAddLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, video, text, eps=1e-5):
        # Save for backward
        ctx.eps = eps
        
        # Ensure contiguous
        video = video.contiguous()
        text = text.contiguous()
        
        # create output buffer
        out = torch.empty_like(video)
        
        if optimulti_fusion_cuda is not None and video.is_cuda:
            optimulti_fusion_cuda.fused_add_layernorm(video, text, out, eps)
        else:
            # Fallback
            out = torch.layer_norm(video + text, video.shape[-1:], eps=eps)
            
        ctx.save_for_backward(out, video, text)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Implementation of backward pass using PyTorch standard ops for stability
        # (Writing a fused backward kernel is significantly more complex)
        out, video, text = ctx.saved_tensors
        eps = ctx.eps
        
        # Recompute Forward components for gradient math
        # LN(x) = (x - mean) / std
        # We need to leverage PyTorch's autograd for the math or derive manually.
        # To keep it simple and correct, we use PyTorch's automatic differentiation 
        # on the equivalent graph for the backward pass.
        # But wait, we are inside specific backward. We must return grads for inputs.
        
        # Let x = video + text
        # y = LayerNorm(x)
        # We need dl/dx. dl/dvideo = dl/dx, dl/dtext = dl/dx.
        
        with torch.enable_grad():
            v_temp = video.detach().requires_grad_(True)
            t_temp = text.detach().requires_grad_(True)
            out_temp = torch.layer_norm(v_temp + t_temp, video.shape[-1:], eps=eps)
            
            grad_temp = torch.autograd.grad(out_temp, (v_temp, t_temp), grad_output, create_graph=False)
            
        return grad_temp[0], grad_temp[1], None

class FusedFusionBlock(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim
        # Projection layer (The "P" in "Normalization + Projection")
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, video_embeds, text_embeds):
        # Apply Custom Fused Add+LN
        # video_embeds: [B, S, D]
        # text_embeds: [B, S, D] (Assumed aligned or expanded)
        
        # Fused Op
        fused = FusedAddLayerNormFunction.apply(video_embeds, text_embeds, self.eps)
        
        # Projection
        return self.projection(fused)

class OptiMultiVideo(nn.Module):
    def __init__(self, vision_encoder, text_decoder, hidden_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.fusion = FusedFusionBlock(hidden_dim)
        
    def forward(self, frames, text_prompt_ids):
        # 1. Vision Features
        # frames: [B, T, C, H, W] -> [B, T, D]
        video_feats = self.vision_encoder(frames)
        
        # 2. Text Features
        # text_prompt_ids: [B, S]
        # We need embeddings. Assuming decoder has embedding access or method.
        # For this demo, let's assume text_decoder returns embeddings for prompt
        text_feats = self.text_decoder.get_embeddings(text_prompt_ids)
        
        # Ensure shapes match for element-wise fusion (Simple Demo)
        # If lengths differ, we truncate or tile. 
        # For specific "Video Captioning" task, usually Cross-Attn is used.
        # But prompt asked for "fused kernel... Element-wise Multi-modal Projection".
        # This implies a specific architecture where V and T are aligned or summed.
        # Let's assume we sum them (like adding positional embeddings).
        
        if video_feats.shape[1] != text_feats.shape[1]:
            # Simple truncation/pad for demo
            min_len = min(video_feats.shape[1], text_feats.shape[1])
            video_feats = video_feats[:, :min_len, :]
            text_feats = text_feats[:, :min_len, :]
            
        # 3. Fuse
        multimodal_feats = self.fusion(video_feats, text_feats)
        
        # 4. Decode (Next token prediction based on fused features?)
        # Or we feed fused features BACK into decoder.
        out_logits = self.text_decoder(multimodal_feats)
        
        return out_logits
