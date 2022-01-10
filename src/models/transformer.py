
import torch
import torch.nn as nn

from vit_pytorch import ViT as VIT_backbone
from vit_pytorch.cait import CaiT


class VIT(nn.Module):

    def __init__(self):

        super().__init__()
        """
        self.backbone = CaiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 64,
            depth = 8,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
        """
        self.backbone = VIT_backbone(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
       
        #self.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)


    def forward(self, inputs):
        x = self.backbone(inputs)
        #x = self.fc(x)
        return x
        