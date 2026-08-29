import torch
import torch.nn as nn
import timm


class ExpertBranch(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PrototypeAttention(nn.Module):
    def __init__(self, dim, num_prototypes=8, num_heads=4):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, D]. Prototypes query the fused morphology feature.
        b = x.size(0)
        q = self.prototypes.unsqueeze(0).expand(b, -1, -1)
        kv = x.unsqueeze(1)
        attended, weights = self.attn(q, kv, kv, need_weights=True)
        pooled = attended.mean(dim=1)
        return self.norm(x + pooled), weights


class OvaMorphNet(nn.Module):
    """Prototype-guided adaptive expert attention network.

    This is the first runnable research prototype. It is intentionally modular so
    that experts, routing, attention, and the backbone can be ablated later.
    """

    def __init__(
        self,
        num_classes,
        backbone_name="efficientnet_b0",
        pretrained=True,
        num_prototypes=8,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        dim = self.backbone.num_features

        self.experts = nn.ModuleList([
            ExpertBranch(dim, dropout),
            ExpertBranch(dim, dropout),
            ExpertBranch(dim, dropout),
        ])
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3),
        )
        self.prototype_attention = PrototypeAttention(
            dim=dim,
            num_prototypes=num_prototypes,
            num_heads=4,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x, return_aux=False):
        features = self.backbone(x)
        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )
        routing_weights = torch.softmax(self.router(features), dim=-1)
        fused = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
        attended, attention_weights = self.prototype_attention(fused)
        logits = self.classifier(attended)

        if return_aux:
            return {
                "logits": logits,
                "routing_weights": routing_weights,
                "prototype_attention": attention_weights,
            }
        return logits
