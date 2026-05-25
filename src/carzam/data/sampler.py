"""Batch samplers used by the contrastive trainer.

The default RandomSampler / SequentialSampler give arbitrary batches — for
SupCon, an anchor needs at least one in-batch peer with the same label. A
batch of 32 unique-class samples produces zero positives and the loss is
identically zero.

BalancedClassBatchSampler fixes this by drawing `n_classes_per_batch`
classes per batch, then `n_samples_per_class` rows from each. With K=8
classes × N=4 samples = 32-batch, every anchor has 3 positives + 28
negatives. That's the geometry SupCon expects.

Re-shuffles each epoch. Skips classes that don't have enough samples.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Sequence

from torch.utils.data import Sampler

from carzam.data.manifest import ManifestRow


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset indices — each list is one batch.

    `key_fn(row) -> hashable` decides the label used to group rows. Default
    keys by `row.car` (the fine class). Pass `lambda r: family_for(r.car)` to
    train a family-level contrastive head instead.
    """

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        n_classes_per_batch: int,
        n_samples_per_class: int,
        steps_per_epoch: int | None = None,
        seed: int = 0,
        key_fn=None,
    ) -> None:
        # newer PyTorch Sampler base takes no args; don't pass data_source.
        super().__init__()
        self.rows = rows
        self.K = n_classes_per_batch
        self.N = n_samples_per_class
        self.batch_size = self.K * self.N
        self.seed = seed
        self.key_fn = key_fn or (lambda r: r.car)

        # Group dataset indices by class label.
        by_class: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_class.setdefault(self.key_fn(r), []).append(i)

        # Need at least N samples to draw a positive group from a class.
        self.by_class: dict[str, list[int]] = {
            k: v for k, v in by_class.items() if len(v) >= self.N
        }
        if len(self.by_class) < self.K:
            raise ValueError(
                f"need ≥ {self.K} classes with at least {self.N} samples each, "
                f"got {len(self.by_class)} qualifying classes. "
                f"Reduce n_classes_per_batch or n_samples_per_class."
            )
        self.classes = sorted(self.by_class.keys())

        # An "epoch" is by default sized so that on average every sample is
        # seen once. Callers can override.
        if steps_per_epoch is None:
            n_samples = sum(len(v) for v in self.by_class.values())
            steps_per_epoch = max(1, n_samples // self.batch_size)
        self.steps_per_epoch = steps_per_epoch

        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Optional — DistributedSampler-style. Lets each epoch reshuffle
        deterministically when given a seed."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        # Per-class shuffled queues — we pop from the front, refill on empty.
        queues: dict[str, list[int]] = {}
        for k in self.classes:
            q = list(self.by_class[k])
            rng.shuffle(q)
            queues[k] = q

        def draw(k: str, n: int) -> list[int]:
            out: list[int] = []
            while len(out) < n:
                if not queues[k]:
                    q = list(self.by_class[k])
                    rng.shuffle(q)
                    queues[k] = q
                out.append(queues[k].pop())
            return out

        for _ in range(self.steps_per_epoch):
            picked = rng.sample(self.classes, self.K)
            batch: list[int] = []
            for k in picked:
                batch.extend(draw(k, self.N))
            rng.shuffle(batch)
            yield batch
