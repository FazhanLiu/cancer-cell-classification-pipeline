import timm
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNTransformerHybrid(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        in_channels = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Linear(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feature_map = self.backbone(x)[-1]
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.classifier(pooled)


def create_model(name: str, num_classes: int = 2, pretrained: bool = True, **kwargs) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if name == "hybrid_cnn_transformer":
        return CNNTransformerHybrid(pretrained=pretrained, num_classes=num_classes, **kwargs)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
