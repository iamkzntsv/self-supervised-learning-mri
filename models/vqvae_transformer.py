import torch
import torch.nn as nn
from generative.networks.nets import DecoderOnlyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self, spatial_shape, latent_dim, attn_layers_dim, attn_layers_depth, attn_layers_heads, embedding_dropout_rate):
        super(Transformer, self).__init__()

        self.transformer = DecoderOnlyTransformer(
            num_tokens=latent_dim + 1,
            max_seq_len=spatial_shape[0] * spatial_shape[1],
            attn_layers_dim=32,
            attn_layers_depth=2,
            attn_layers_heads=2,
            embedding_dropout_rate=0
        )

    def forward(self, x):
        return self.transformer(x)
