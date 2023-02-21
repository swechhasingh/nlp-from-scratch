import torch.nn as nn
import torch
import torch.nn.functional as F

# Let's modify our baseline bigram neural netwrok model with position embedding layer
class EncDecTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        block_size,
        model_dim,
        n_layer,
        n_head,
        cross_attention=True,
    ) -> None:
        super().__init__()
        self.block_size = block_size  # max context length
        self.model_dim = model_dim
        # input: (B, T) output: (B, T, model_dim)
        self.src_embed = nn.Embedding(src_vocab_size, model_dim)
        self.target_embed = nn.Embedding(tgt_vocab_size, model_dim)
        # input: (T,) output: (T, model_dim)
        self.pos_embed_layer = nn.Embedding(block_size, model_dim)

        self.encoder = Encoder(n_head, model_dim, n_layer)
        self.decoder = Decoder(n_head, model_dim, n_layer, cross_attention)

        self.decoded_target_proj = nn.Linear(model_dim, tgt_vocab_size)

    def encode(self, x, mask):

        inp_token_emb = self.src_embed(x)
        inp_pos_emb = self.pos_embed_layer(
            torch.arange(self.block_size, device=x.device)
        )
        # broadcast pos_emb along the batch dimension of token_emb
        inp_token_emb = inp_token_emb + inp_pos_emb

        return self.encoder(inp_token_emb, mask)

    def decode(self, target, target_mask, memory, src_mask):
        token_emb = self.target_embed(target)
        pos_emb = self.pos_embed_layer(
            torch.arange(target.shape[-1], device=target.device)
        )
        # broadcast pos_emb along the batch dimension of token_emb
        token_emb = token_emb + pos_emb

        return self.decoder(token_emb, target_mask, memory, src_mask)

    def forward(self, src, src_mask, target=None, target_mask=None, target_y=None):

        memory = self.encode(src, src_mask)
        dec_out = self.decode(target, target_mask, memory, src_mask)
        logits = self.decoded_target_proj(dec_out)

        if target_y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.contiguous().view(B * T, C)
            target_y = target_y.contiguous().view(B * T)
            loss = F.cross_entropy(logits, target_y)
        return logits, loss

    def generate(self, src, src_mask, max_tokens=15, sos=0, eos=1):
        memory = self.encode(src, src_mask)
        tgt = torch.zeros(1, 1, dtype=src.dtype, device=src.device)

        for i in range(max_tokens):
            # causal attention mask: (1,T,T)
            tgt_mask = torch.tril(
                torch.ones(
                    1,
                    1,
                    tgt.shape[-1],
                    tgt.shape[-1],
                    dtype=torch.bool,
                    device=tgt.device,
                )
            )
            # block_size is the maximum context length of our LM, therefore we can only use last block_size characters as input to generate next character
            dec_out = self.decode(tgt, tgt_mask, memory, src_mask)
            logits = self.decoded_target_proj(dec_out)
            # dec_inp: (B,T_dec) logits: (B, T_dec, vocab_size)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            next_inp = torch.multinomial(probs, num_samples=1)
            print(tgt.shape, next_inp.shape)
            tgt = torch.cat((tgt, next_inp), dim=1)

            if next_inp.item() == eos:
                break
        return tgt


class Encoder(nn.Module):
    def __init__(self, n_head, model_dim, n_layer) -> None:
        super().__init__()
        # Stacked Transformer Encoder Blocks
        self.layers = nn.Sequential(
            *[TransformerEncoderBlock(n_head, model_dim) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, n_layer, cross_attention) -> None:
        super().__init__()
        # Stacked Transformer Encoder Blocks
        self.layers = nn.Sequential(
            *[
                TransformerDecoderBlock(n_head, model_dim, cross_attention)
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, target_mask, memory=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, target_mask, memory, src_mask)
        return self.norm(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, model_dim) -> None:
        super().__init__()
        self.n_head = n_head
        self.multi_head_layer = MultiHeadAttention(n_head, model_dim)
        self.mh_layer_norm = nn.LayerNorm(model_dim)

        self.ff_layer = PositionWiseFeedForwardNetwork(model_dim)
        self.ff_layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        # LayerNorm(x + Sublayer(x))
        x_norm = self.mh_layer_norm(x)
        x = x + self.multi_head_layer(x_norm, x_norm, x_norm, mask)
        x = x + self.ff_layer(self.ff_layer_norm(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_head, model_dim, cross_attention=False) -> None:
        super().__init__()
        self.n_head = n_head
        self.cross_attention = cross_attention
        self.self_attn_layer = MultiHeadAttention(n_head, model_dim)
        self.self_attn_layer_norm = nn.LayerNorm(model_dim)

        if self.cross_attention:
            self.cross_attn_layer = MultiHeadAttention(n_head, model_dim)
            self.cross_attn_layer_norm = nn.LayerNorm(model_dim)

        self.ff_layer = PositionWiseFeedForwardNetwork(model_dim)
        self.ffn_layer_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x,
        tgt_mask,
        memory=None,
        src_mask=None,
    ):
        x_norm = self.self_attn_layer_norm(x)
        x = x + self.self_attn_layer(x_norm, x_norm, x_norm, tgt_mask)
        if self.cross_attention:
            x = x + self.cross_attn_layer(
                self.cross_attn_layer_norm(x),  # query
                memory,  # key
                memory,  # value
                src_mask,
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
    def __init__(self, n_head, model_dim) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        # parallelized multi-head attention implementation using matrix multiplication
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)

        self.linear_proj = nn.Linear(model_dim, model_dim)

        self.attention = None

    def forward(self, query, key, value, mask=None):
        # query (input): (B,T,model_dim), query (output): (B,T,model_dim)
        # rearrange query output in (B,T,n_head,head_dim) shape
        # further transpose query to (B,n_head,T,head_dim) shape
        B, T, model_dim = query.shape
        query = (
            self.query_proj(query)
            .view(B, T, self.n_head, self.head_dim)
            .transpose(1, 2)
        )
        key = self.key_proj(key).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        value = (
            self.value_proj(value)
            .view(B, T, self.n_head, self.head_dim)
            .transpose(1, 2)
        )

        # head_out: (B,T,n_head,head_dim)
        print(mask.shape)
        head_out, self.attention = self.scaled_dot_product_attn(key, value, query, mask)
        print(head_out.shape)
        # concatenate outputs from all the heads
        # head_out: (B,T,model_dim)
        head_out = head_out.contiguous().view(B, T, self.n_head * self.head_dim)
        print(head_out.shape)
        # proj output: (B,T,model_dim)
        return self.linear_proj(head_out)

    @staticmethod
    def scaled_dot_product_attn(Q, K, V, mask=None):
        head_dim = Q.shape[-1]
        # scaled_dot_product_attention
        # MatMul
        # before matrix multiplication, transpose K(key) to (B,n_head,head_dim,T)
        weight_matrix = Q @ K.transpose(-2, -1)
        # weight_matrix:(B,n_head,T,T)
        # Scale
        weight_matrix = weight_matrix * head_dim ** (-0.5)
        # apply mask, mask:(B,1,T,T) or (B,1,1,T)
        weight_matrix = weight_matrix.masked_fill(mask == 0, float("-inf"))
        # apply softmax on last dim
        weight_matrix = F.softmax(weight_matrix, dim=-1)
        # MatMul (weighted sum of V to get a representation for every input location)
        # V: (B,n_head,T,head_dim), out: (B,n_head,T,head_dim)
        out = weight_matrix @ V
        # out: (B,T,n_head,head_dim)
        return out.transpose(1, 2), weight_matrix
