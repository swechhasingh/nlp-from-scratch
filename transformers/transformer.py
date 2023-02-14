import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, n_head, model_dim, block_size, causal_attention) -> None:
        super().__init__()
        self.n_head = n_head
        self.multi_head_layer = MultiHeadAttention(
            n_head, model_dim, block_size, causal_attention
        )
        self.head_layer_norm = nn.LayerNorm(model_dim)

        self.ff_layer = PositionWiseFeedForwardNetwork(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x + self.multi_head_layer(self.head_layer_norm(x))
        x = x + self.ff_layer(self.ffn_layer_norm(x))
        return x


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, model_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, block_size, causal_attention) -> None:
        super().__init__()
        self.head_dim = model_dim // n_head
        self.heads = nn.ModuleList(
            [
                Head(self.head_dim, model_dim, block_size, causal_attention)
                for _ in range(n_head)
            ]
        )
        self.linear_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        head_out = torch.cat([head(x) for head in self.heads], dim=-1)  # parallelizable
        return self.linear_proj(head_out)


class Head(nn.Module):
    def __init__(self, head_dim, model_dim, block_size, causal_attention=False) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.block_size = block_size
        self.causal_attention = causal_attention
        # for each head, apply different learned linear transform on x to get query, key and value
        self.query = nn.Linear(model_dim, head_dim, bias=False)
        self.key = nn.Linear(model_dim, head_dim, bias=False)
        self.value = nn.Linear(model_dim, head_dim, bias=False)
        # for causal attention
        if causal_attention:
            self.register_buffer(
                "tril", torch.tril(torch.ones(block_size, block_size))
            )  # max context length = block_size

    def forward(self, x):
        B, T, C = x.shape  # T is sequence length
        Q = self.query(x)  # (batch_size, T, head_dim)
        K = self.key(x)  # (batch_size, T, head_dim)
        V = self.value(x)  # (batch_size, T, head_dim)

        # scaled_dot_product_attention
        # MatMul
        weight_matrix = Q @ K.transpose(1, 2)
        # Scale
        weight_matrix = weight_matrix * self.head_dim ** (-0.5)
        # Mask for causal self-attention
        if self.causal_attention:
            # mask: self.tril[:T, :T]
            weight_matrix = weight_matrix.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )
        # softmax; weight_matrix:(batch_size, T, T)
        weight_matrix = F.softmax(weight_matrix, dim=-1)
        # MatMul (weighted sum)
        out = weight_matrix @ V  # out: (batch_size, T, head_dim)
        return out
