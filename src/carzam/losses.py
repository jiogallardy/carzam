"""Supervised contrastive loss.

Khosla et al. 2020 ("Supervised Contrastive Learning"). The key difference vs
SimCLR's NT-Xent is that we use labels at training time: for each anchor,
*all* other in-batch samples sharing its label are positives, not just one
augmented twin. This handles K-cars × N-samples-each batches naturally.

For each anchor i with label y_i:
    L_i = -1/|P(i)| * Σ_{p in P(i)} log [ exp(z_i · z_p / τ) / Σ_{a≠i} exp(z_i · z_a / τ) ]

where P(i) = { j ≠ i : y_j == y_i }, τ is a temperature (we use 0.07 from the
paper). Anchors with no in-batch positives are skipped (loss contribution 0).

Embeddings are expected to be L2-normalized — CarAudioModel's embedding head
already normalizes its output, so we can just dot-product to get cosine sim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,   # (B, D), L2-normalized
    labels: torch.Tensor,       # (B,) int64
    temperature: float = 0.07,
) -> torch.Tensor:
    """Returns a scalar loss. Mean over anchors that have ≥1 same-class peer."""
    if embeddings.dim() != 2 or labels.dim() != 1:
        raise ValueError(
            f"expected embeddings (B, D) and labels (B,), got "
            f"{tuple(embeddings.shape)} and {tuple(labels.shape)}"
        )
    b = embeddings.shape[0]
    device = embeddings.device

    # Cosine sim matrix (B, B). Diagonal is 1.0 because vectors are unit-norm.
    sim = embeddings @ embeddings.T / temperature

    # Mask out self-similarity from both numerator and denominator.
    self_mask = torch.eye(b, device=device, dtype=torch.bool)

    # Positive mask: same label, not self.
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_eq & ~self_mask

    # Numerical stability: subtract per-row max before exp.
    # This is the standard log-sum-exp trick; doesn't change the loss value.
    sim_max, _ = sim.masked_fill(self_mask, float("-inf")).max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()

    exp_logits = torch.exp(logits)
    # Denominator: sum over all j ≠ i.
    denom = exp_logits.masked_fill(self_mask, 0.0).sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denom + 1e-12)

    # Average log-prob over positives per anchor; anchors with no positives
    # contribute 0 (we'd otherwise divide by zero).
    n_positives = pos_mask.sum(dim=1)
    valid = n_positives > 0
    if not valid.any():
        # No usable anchors in this batch — return a zero loss that still
        # participates in autograd so the trainer doesn't blow up.
        return embeddings.sum() * 0.0
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    per_anchor = pos_log_prob[valid] / n_positives[valid].float()
    return -per_anchor.mean()
