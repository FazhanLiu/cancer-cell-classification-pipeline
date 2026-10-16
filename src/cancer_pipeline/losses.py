import math

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha_negative=0.5, alpha_positive=0.5, label_smoothing=0.0):
        super().__init__()
        if not all(math.isfinite(v) for v in (gamma, alpha_negative, alpha_positive, label_smoothing)) or gamma < 0 or min(alpha_negative, alpha_positive) < 0 or alpha_negative + alpha_positive <= 0 or not 0 <= label_smoothing < 1:
            raise ValueError('Invalid focal loss parameters.')
        self.gamma, self.label_smoothing = gamma, label_smoothing
        self.register_buffer('alpha', torch.tensor([alpha_negative, alpha_positive], dtype=torch.float32))

    def loss_terms(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        target = F.one_hot(targets, logits.shape[1]).to(logits.dtype)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / logits.shape[1]
        ce = -(target * log_probs).sum(1)
        pt = (target * log_probs.exp()).sum(1)
        return (self.alpha[targets] * (1 - pt).pow(self.gamma) * ce).sum()

    def denominator(self, targets):
        return targets.numel()

    def forward(self, logits, targets):
        return self.loss_terms(logits, targets) / self.denominator(targets)


def loss_denominator(criterion, targets):
    if isinstance(criterion, nn.CrossEntropyLoss):
        if criterion.reduction != 'mean' or criterion.ignore_index in (0, 1):
            raise ValueError('Training requires mean CE without ignored binary classes.')
        return float(criterion.weight[targets].sum().item()) if criterion.weight is not None else targets.numel()
    return float(criterion.denominator(targets))


def loss_numerator(criterion, logits, targets):
    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(logits, targets, weight=criterion.weight, reduction='sum', label_smoothing=criterion.label_smoothing)
    return criterion.loss_terms(logits, targets)
