import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_frames=8, image_size=224, patch_size=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Simple Patch Embedding: (B, T, C, H, W) -> (B, T, Tokens, D) -> Flatten (B, Seq, D)
        # For simplicity in this demo: Project entire frame to one vector or use minimal patches
        # Let's do a linear projection of flattened frame for "Video Token"
        input_dim = 3 * image_size * image_size
        # Actually, let's just project meaningful random noise for the architecture demo
        # or a real ResNet backbone would be too heavy
        # Let's simulate a pre-computed feature extractor input
        # Input: [B, T, C, H, W]
        
        self.patch_embed = nn.Sequential(
            nn.Conv3d(3, hidden_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size)),
            nn.Flatten(2) 
        )
        # After convert: [B, D, T, H', W'] -> need specific reshaping
        
    def forward(self, x):
        # x: [B, T=8, C=3, H=224, W=224]
        # Demo simplification: map directly to [B, T, D] via a "Global Pool" simulation
        # Real impl would be ViT.
        
        b, t, c, h, w = x.shape
        # Simplified: Treat each frame as a token
        # Flatten frame
        x_flat = x.view(b, t, -1)
        # We need a linear layer to project to hidden_dim if we process raw pixels
        # But for 'OptiMulti', let's assume raw pixels input and we have a Linear
        if not hasattr(self, 'projector'):
            self.projector = nn.Linear(c*h*w, self.hidden_dim).to(x.device)
            
        return self.projector(x_flat) # [B, T, D]
