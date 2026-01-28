import torch
import torch.nn as nn

class TextDecoder(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=768, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def get_embeddings(self, token_ids):
        b, s = token_ids.shape
        pos = torch.arange(s, device=token_ids.device).unsqueeze(0)
        return self.embedding(token_ids) + self.pos_emb(pos)

    def forward(self, fused_feats):
        # fused_feats: [B, Seq, D] acts as "memory" or pure state?
        # In a real captioner, we attend to visual features.
        # Here we updated fused_feats which ARE the multimodal state.
        # So we just decode them to logits.
        
        # Simple causal decode or just MLM for demo?
        # Let's just pass through decoder layers self-attending
        out = self.decoder(fused_feats, fused_feats) 
        return self.head(out)
