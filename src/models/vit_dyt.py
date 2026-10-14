import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from logs.cmd.logger_config import setup_logger
from src.utils.models_extensions import PatchEmbedding, TransformerEncoderBlockDyT, DyT

LOG_NAME = "vit_dyt_log"
logger = setup_logger(LOG_NAME, save_to_file=False)


class ViT_DyT(nn.Module):
    """
    Vision Transformer Architecture using DyT (Dynamic Tanh).
    This is the experimental model based on the paper
    "Transformers without Normalization" (2025).
    All hyperparameters are identical to ViT_LayerNorm for experimental isolation.
    """

    def __init__(
        self,
        img_channels=3,
        patch_size=4,
        embed_dim=256,
        depth=6,  # number of Transformer blocks
        # ViT-Lite configuration for small datasets (Lee et al., 2021) to prevent overfitting
        num_heads=8,  # number of attention heads
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()

        # 1. extraction and Embedding of Patches (Reaproveitado do Utils!)
        self.patch_embed = PatchEmbedding(img_channels, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. class Token (Trainable) and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. transformer blocks stack (encoder) using DyT
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlockDyT(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # 4. final dynamic normalization and linear head
        self.norm = DyT(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # customized initialization of weights for faster convergence
        self._init_weights()

    def _init_weights(self):
        # normal initialization for embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Xavier initialization for linear layers and correct init for DyT
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # especific initialization for the Dynamic Tanh (DyT)
            elif isinstance(m, DyT):
                nn.init.constant_(m.beta, 0)
                nn.init.constant_(m.gamma, 1.0)
                nn.init.constant_(m.alpha, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]

        # pass the image through the PatchEmbedding
        x = self.patch_embed(x)

        # expands the cls token for the entire batch
        # and appends it to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # adds the positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # pass the sequence through all the Transformer blocks
        for block in self.blocks:
            x = block(x)

        # apply the final normalization (DyT)
        x = self.norm(x)

        # for classification, we use only the output of the [CLS] token (index 0)
        cls_output = x[:, 0]

        # generate the logits (raw predictions) for the 10 classes of CIFAR-10
        logits = self.head(cls_output)

        return logits


def vit_dyt_dummy_test():
    """
    Simulates a batch of CIFAR-10: 4 images, 3 channels, 32x32 pixels
    """
    logger.info("Running dummy test for ViT_DyT...")

    dummy_input = torch.randn(4, 3, 32, 32)
    model = ViT_DyT()

    output = model(dummy_input)

    logger.info(f"Input Shape: {dummy_input.shape}")
    logger.info(f"Output Shape: {output.shape} (Expected: [4, 10] for the CIFAR-10 classes)")


if __name__ == "__main__":
    testing: bool = False

    if testing:
        vit_dyt_dummy_test()
