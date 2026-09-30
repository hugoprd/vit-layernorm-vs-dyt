# import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts 2D images into a sequence of 1D patches.
    For CIFAR-10 (32x32) with patch_size=4, it's generated an 8x8 grid of patches.
    Total number of patches = 64.
    """

    def __init__(self, img_channels=3, patch_size=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2  # 8 * 8 = 64

        # Conv2d it's the more efficient way to extract patches without overlap
        # and already project them to the embedding dimension
        # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
        self.proj = nn.Conv2d(
            in_channels=img_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # Original shape: (batch_size, img_channels, 32, 32)
        x = self.proj(x)
        # Output shape: (batch_size, embed_dim, 8, 8)

        x = x.flatten(
            2
        )  # flatten the spatial dimensions (H, W) into a single sequence dimension
        # Output shape: (batch_size, embed_dim, 64)

        # Transpose to the expected format for the
        # Transformer: (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    Common block of the Transformer using Pre-Layer Normalization (Pre-LN).
    The normalization is applied BEFORE the attention and the MLP.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 1. Attention Block: Normalizes -> Applies Attention -> Residual Connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # 2. MLP Block: Normalizes -> Applies MLP -> Residual Connection
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x
