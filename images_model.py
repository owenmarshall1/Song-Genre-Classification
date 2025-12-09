import torch
import torch.nn as nn
import math
import torchvision.models as models
import torch
import numpy as np 
import timm

##### custom Vision Transformer (ViT)
class ImageViT(nn.Module):
    def __init__( self, num_classes, img_size=224, patch_size=16, emb_dim=256, depth=6, num_heads=8, mlp_dim=512, dropout=0.1 ): 
        super().__init__()

        # get number of patches
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        #Linear layer for pathc embedding 
        self.patch_embed = nn.Linear(patch_dim, emb_dim)
        
        # MLP block
        self.patch_norm = nn.LayerNorm(emb_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(dropout)
        )
        # cls token used for the image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        
        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # final head 
        self.pos_drop = nn.Dropout(dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

        self.patch_size = patch_size
        
        # initialize learnable parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

#### function to split image into patches
    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p * p)
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(2)
        return x
    
#### forward pass 
    def forward(self, x):
        B = x.size(0)
        patches = self._patchify(x)
        tokens = self.patch_embed(patches)

        tokens = self.patch_norm(tokens)
        tokens = tokens + self.patch_mlp(tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]

        return self.mlp_head(cls_out)

##### pretrained DeiT model with custom head
class Deit(nn.Module):
    def __init__(self,num_classes,):

        super().__init__()

        # load pretrained DeiT-tiny from timm
        self.vit = timm.create_model("deit_tiny_patch16_224",pretrained= True, num_classes= 0)
        feature = self.vit.num_features

        # custom head layer 
        self.head = nn.Sequential(
            nn.Linear(feature, feature//2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(feature//2, num_classes)
         )
    #forward pass
    def forward(self , X):
        X = self.vit(X)
        X = self.head(X)
        return X
