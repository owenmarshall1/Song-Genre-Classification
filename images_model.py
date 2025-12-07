import torch
import torch.nn as nn
import math

class ImageViT(nn.Module):
    def __init__(
            self,
            num_classes,
            img_size=128,
            patch_size=16,
            emb_dim=256,
            depth=6,
            num_heads=8,
            mlp_dim=512,
            dropout=0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, emb_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

        self.patch_size = patch_size

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p * p)
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(2)
        return x
    
    def forward(self, x):
        B = x.size(0)
        patches = self._patchify(x)
        tokens = self.patch_embed(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]

        return self.mlp_head(cls_out)