"""
Classification and regression heads for fine-tuning a pretrained transformer
on microbiome benchmark tasks.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

PoolingStrategy = Literal["mean", "first_token", "last_token", "cls_token"]


def _mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1).long() - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(
        hidden_states.size(0), device=hidden_states.device, dtype=torch.long
    )
    return hidden_states[batch_idx, last_idx]


def _pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: PoolingStrategy,
) -> torch.Tensor:
    if strategy == "mean":
        return _mean_pool(hidden_states, attention_mask)
    if strategy in ("first_token", "cls_token"):
        return hidden_states[:, 0]
    if strategy == "last_token":
        return _last_token_pool(hidden_states, attention_mask)
    raise ValueError(f"Unknown pooling strategy: {strategy}")


class ClassificationModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        label_dims: list[int],
        pooling_strategy: PoolingStrategy,
        drug_dim: int = 0,
        class_weights: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_labels = len(label_dims)
        self.label_dims = label_dims
        self.drug_dim = drug_dim
        hidden = base_model.config.hidden_size + drug_dim
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in label_dims])
        self.pooling_strategy: PoolingStrategy = pooling_strategy
        self.class_weights = class_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug_onehot: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = _pool(hidden_states, attention_mask, self.pooling_strategy)
        if drug_onehot is not None:
            pooled = torch.cat([pooled, drug_onehot], dim=-1)

        logits_per_head = [head(pooled) for head in self.heads]
        logits_padded = pooled.new_full(
            (pooled.size(0), self.num_labels, max(self.label_dims)),
            float("-inf"),
        )
        for h, logits_h in enumerate(logits_per_head):
            logits_padded[:, h, : self.label_dims[h]] = logits_h

        loss = None
        if labels is not None:
            total_loss = pooled.new_zeros(())
            total_count = 0
            for h in range(self.num_labels):
                mask_h = labels[:, h] != -100
                n_valid = mask_h.sum().item()
                if n_valid > 0:
                    weight = None
                    if self.class_weights is not None:
                        weight = self.class_weights[h].to(logits_per_head[h].device)
                    total_loss += (
                        nn.functional.cross_entropy(
                            logits_per_head[h][mask_h],
                            labels[mask_h, h].long(),
                            reduction="mean",
                            weight=weight,
                        )
                        * n_valid
                    )
                    total_count += n_valid
            loss = total_loss / total_count

        return {"loss": loss, "logits": logits_padded}


class RegressionModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_targets: int,
        pooling_strategy: PoolingStrategy,
        drug_dim: int = 0,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_targets = num_targets
        self.drug_dim = drug_dim
        hidden = base_model.config.hidden_size + drug_dim
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_targets)])
        self.pooling_strategy: PoolingStrategy = pooling_strategy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug_onehot: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = _pool(hidden_states, attention_mask, self.pooling_strategy)
        if drug_onehot is not None:
            pooled = torch.cat([pooled, drug_onehot], dim=-1)
        preds = torch.cat([head(pooled) for head in self.heads], dim=-1)

        loss = None
        if targets is not None:
            targets = targets.float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)
            loss = nn.functional.mse_loss(preds, targets)

        return {"loss": loss, "logits": preds.squeeze(-1)}
