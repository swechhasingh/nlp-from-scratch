import torch.nn as nn
import torch
import torch.nn.functional as F

# Let's modify our baseline bigram neural netwrok model with position embedding layer
class Transformer(nn.Module):
    def __init__(
        self, vocab_size, block_size, model_dim, n_layer, n_head, causal_attention
    ) -> None:
        super().__init__()
        self.block_size = block_size  # max context length
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.token_embed_layer = nn.Embedding(
            vocab_size, model_dim
        )  # input: (B, T) output: (B, T, model_dim)
        self.pos_embed_layer = nn.Embedding(
            block_size, model_dim
        )  # input: (T,) output: (T, model_dim)
        # Stacked Transformer Blocks
        self.layers = nn.Sequential(
            *[
                TransformerEncoderBlock(n_head, model_dim, block_size, causal_attention)
                for _ in range(n_layer)
            ]
        )

        self.out_linear_proj = nn.Linear(model_dim, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape  # idx: (B, T) T is sequence length
        token_emb = self.token_embed_layer(idx)
        pos_emb = self.pos_embed_layer(torch.arange(T, device=idx.device))
        # broadcast pos_emb along the batch dimension of token_emb
        token_emb = token_emb + pos_emb

        logits = self.out_linear_proj(self.layers(token_emb))

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_token):

        for i in range(max_token):
            # block_size is the maximum context length of our LM, therefore we can only use last block_size characters as input to generate next character
            logits, _ = self(
                idx[:, -self.block_size :]
            )  # idx: (B,T) logits: (B, T, vocab_size)
            logits = logits[
                :, -1, :
            ]  # only last position token is required to generate next character
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            sample = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sample), dim=1)
        return idx


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, model_dim, block_size) -> None:
        super().__init__()
        self.n_head = n_head
        self.multi_head_layer = MultiHeadAttention(
            n_head, model_dim, block_size, causal_attention=False
        )
        self.head_layer_norm = nn.LayerNorm(model_dim)

        self.ff_layer = PositionWiseFeedForwardNetwork(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x + self.multi_head_layer(self.head_layer_norm(x))
        x = x + self.ff_layer(self.ffn_layer_norm(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_head, model_dim, block_size, cross_attention=False) -> None:
        super().__init__()
        self.n_head = n_head
        self.cross_attention = cross_attention
        self.multi_head_self_attn_layer = MultiHeadAttention(
            n_head, model_dim, block_size, causal_attention=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(model_dim)

        if self.cross_attention:
            self.multi_head_cross_attn_layer = MultiHeadAttention(
                n_head, model_dim, block_size, causal_attention=False
            )
            self.cross_attn_layer_norm = nn.LayerNorm(model_dim)

        self.ff_layer = PositionWiseFeedForwardNetwork(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output=None):
        x = x + self.multi_head_self_attn_layer(self.self_attn_layer_norm(x))
        if self.cross_attention:
            x = x + self.multi_head_cross_attn_layer(
                self.cross_attn_layer_norm(x),
                self.cross_attn_layer_norm(encoder_output),
            )
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
        # TODO:  # parallelizable
        self.heads = nn.ModuleList(
            [
                Head(self.head_dim, model_dim, block_size, causal_attention)
                for _ in range(n_head)
            ]
        )
        self.linear_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, encoder_output=None):
        # TODO:  # parallelizable
        head_out = torch.cat([head(x, encoder_output) for head in self.heads], dim=-1)
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

    def forward(self, x, encoder_output=None):
        B, T, C = x.shape  # T is sequence length
        Q = self.query(x)  # (batch_size, T, head_dim)
        if encoder_output is None:
            K = self.key(x)  # (batch_size, T, head_dim)
            V = self.value(x)  # (batch_size, T, head_dim)
        else:
            K = self.key(encoder_output)  # (batch_size, T, head_dim)
            V = self.value(encoder_output)  # (batch_size, T, head_dim)

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
