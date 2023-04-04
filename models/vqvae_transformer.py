import torch
from generative.networks.nets import DecoderOnlyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transformer_model(spatial_shape):
    transformer_model = DecoderOnlyTransformer(
        num_tokens=256 + 1,  # 256 from num_embeddings input of VQVAE + 1 for Begin of Sentence (BOS) token
        max_seq_len=spatial_shape[0] * spatial_shape[1],
        attn_layers_dim=64,
        attn_layers_depth=6,
        attn_layers_heads=4,
    )
    return transformer_model
