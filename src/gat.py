"""
GAT Model & Relational Graph Architecture for Name That LLM
Used by EXP_40, EXP_44, EXP_45, and final inference kernels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptGATLayer(nn.Module):
    def __init__(self, d_in, d_out, n_heads=4, dropout=0.15):
        super().__init__()
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.skip_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.ffn = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.Dropout(dropout)
        )
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, K, _ = x.shape
        q = self.q_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, self.d_out)
        h = self.norm1(self.skip_proj(x) + self.out_proj(attn_out))
        out = self.norm2(h + self.ffn(h))
        return out


class PromptResidualGAT(nn.Module):
    def __init__(self, in_features=1296, hidden_dim=128, num_classes=18, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gat1 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.gat2 = PromptGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.gat1(h, mask=mask)
        h = self.gat2(h, mask=mask)
        logits = self.classifier(h)
        return logits


def compute_prompt_residuals(X, prompt_labels):
    """
    Computes r_i = x_i - mean_{j in prompt(i)}(x_j) strictly within each prompt group.
    """
    R = np.zeros_like(X)
    for p in np.unique(prompt_labels):
        mask = (prompt_labels == p)
        mean_p = np.mean(X[mask], axis=0, keepdims=True)
        R[mask] = X[mask] - mean_p
    return R


def pack_prompt_graph_batch(node_features, prompt_labels, node_targets=None, max_k=18):
    """
    Packs variable-sized prompt groups into batched tensors:
      features: (num_prompts, max_k, D)
      mask:     (num_prompts, 1, max_k, max_k)
      targets:  (num_prompts, max_k)
      mapping:  list of (prompt_id, array_of_original_indices)
    """
    unique_prompts = np.unique(prompt_labels)
    n_prompts = len(unique_prompts)
    D = node_features.shape[1]

    batch_x = np.zeros((n_prompts, max_k, D), dtype=np.float32)
    batch_mask = np.zeros((n_prompts, 1, max_k, max_k), dtype=bool)
    batch_y = np.full((n_prompts, max_k), -1, dtype=np.int64)
    batch_valid = np.zeros((n_prompts, max_k), dtype=bool)
    index_mapping = []

    for p_idx, p in enumerate(unique_prompts):
        orig_indices = np.where(prompt_labels == p)[0]
        k = min(len(orig_indices), max_k)
        batch_x[p_idx, :k] = node_features[orig_indices[:k]]
        batch_mask[p_idx, 0, :k, :k] = True
        batch_valid[p_idx, :k] = True
        if node_targets is not None:
            batch_y[p_idx, :k] = node_targets[orig_indices[:k]]
        index_mapping.append((p, orig_indices[:k]))

    return (
        torch.tensor(batch_x, dtype=torch.float32),
        torch.tensor(batch_mask, dtype=torch.bool),
        torch.tensor(batch_y, dtype=torch.long),
        torch.tensor(batch_valid, dtype=torch.bool),
        index_mapping
    )
