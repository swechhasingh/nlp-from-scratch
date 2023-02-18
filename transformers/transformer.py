import torch.nn as nn
import torch
import torch.nn.functional as F

# Let's modify our baseline bigram neural netwrok model with position embedding layer
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        model_dim,
        n_layer,
        n_head,
        cross_attention=True,
    ) -> None:
        super().__init__()
        self.block_size = block_size  # max context length
        self.model_dim = model_dim
        self.vocab_size = vocab_size  # total uniques words in src and target language
        self.token_embed_layer = nn.Embedding(
            vocab_size, model_dim
        )  # input: (B, T) output: (B, T, model_dim)
        self.pos_embed_layer = nn.Embedding(
            block_size, model_dim
        )  # input: (T,) output: (T, model_dim)
        # Stacked Transformer Encoder Blocks
        self.encoder_layers = nn.Sequential(
            *[
                TransformerEncoderBlock(n_head, model_dim, block_size)
                for _ in range(n_layer)
            ]
        )

        self.decoder_layers = nn.Sequential(
            *[
                TransformerDecoderBlock(n_head, model_dim, block_size, cross_attention)
                for _ in range(n_layer)
            ]
        )

        self.out_linear_proj = nn.Linear(model_dim, vocab_size)

    def forward(self, enc_inp, dec_inp, enc_inp_mask=None, dec_target=None):
        B, T_enc = enc_inp.shape  # idx: (B, T) T is sequence length
        inp_token_emb = self.token_embed_layer(enc_inp)
        inp_pos_emb = self.pos_embed_layer(torch.arange(T_enc, device=enc_inp.device))
        # broadcast pos_emb along the batch dimension of token_emb
        inp_token_emb = inp_token_emb + inp_pos_emb

        B, T_dec = dec_inp.shape  # idx: (B, T) T is sequence length
        dec_token_emb = self.token_embed_layer(dec_inp)
        dec_pos_emb = self.pos_embed_layer(torch.arange(T_dec, device=dec_inp.device))
        # broadcast pos_emb along the batch dimension of token_emb
        dec_token_emb = dec_token_emb + dec_pos_emb

        for layer in self.encoder_layers:
            encoder_output = layer(inp_token_emb, enc_inp_mask)
            inp_token_emb = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(dec_token_emb, encoder_output, enc_inp_mask)
            dec_token_emb = decoder_output
        logits = self.out_linear_proj(decoder_output)

        if dec_target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            dec_target = dec_target.contiguous().view(B * T)
            loss = F.cross_entropy(logits, dec_target)
        return logits, loss

    def generate(self, enc_inp, dec_inp, max_token, enc_inp_mask=None):
        EOS_token = 1
        for i in range(max_token):
            # block_size is the maximum context length of our LM, therefore we can only use last block_size characters as input to generate next character
            logits, _ = self(enc_inp, dec_inp, enc_inp_mask)
            # dec_inp: (B,T_dec) logits: (B, T_dec, vocab_size)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            dec_out = torch.multinomial(probs, num_samples=1)
            # print(dec_inp.shape, dec_out.shape)
            dec_inp = torch.cat((dec_inp, dec_out), dim=1)
            if dec_out.item() == EOS_token:
                return dec_inp
        return dec_inp


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

    def forward(self, x, enc_inp_mask=None):
        x = x + self.multi_head_layer(
            self.head_layer_norm(x), enc_inp_mask=enc_inp_mask
        )
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

    def forward(self, x, encoder_output=None, enc_inp_mask=None):
        x = x + self.multi_head_self_attn_layer(self.self_attn_layer_norm(x))
        if self.cross_attention:
            x = x + self.multi_head_cross_attn_layer(
                self.cross_attn_layer_norm(x),
                self.cross_attn_layer_norm(encoder_output),
                enc_inp_mask,
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

    def forward(self, x, encoder_output=None, enc_inp_mask=None):
        # TODO:  # parallelizable
        head_out = torch.cat(
            [head(x, encoder_output, enc_inp_mask) for head in self.heads], dim=-1
        )
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

    def forward(self, x, encoder_output=None, enc_inp_mask=None):
        B, T, C = x.shape  # T is sequence length
        Q = self.query(x)  # (batch_size, T, head_dim)
        # self-attention
        if encoder_output is None:
            K = self.key(x)  # (batch_size, T, head_dim)
            V = self.value(x)  # (batch_size, T, head_dim)
        else:  # cross-attention
            # key and value from encoder output
            K = self.key(encoder_output)  # (batch_size, T, head_dim)
            V = self.value(encoder_output)  # (batch_size, T, head_dim)

        """
        in case of self/causal-attention:
            weight_matrix:(batch_size, T, T)
        in case of cross-attention:
            weight_matrix:(batch_size, T_dec, T_enc)
        """
        # scaled_dot_product_attention
        # MatMul
        weight_matrix = Q @ K.transpose(1, 2)
        # Scale
        weight_matrix = weight_matrix * self.head_dim ** (-0.5)
        # Mask for causal self-attention
        if self.causal_attention:
            # mask: self.tril[:T, :T], broadcasted along batch axis
            weight_matrix = weight_matrix.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )
        # in case of self-attention and cross-attention apply mask for variable length input sequences
        if enc_inp_mask is not None:
            # enc_inp_mask: (B, T) -> (B,1,T)
            B, T_enc = enc_inp_mask.shape
            enc_inp_mask = enc_inp_mask.view(B, 1, T_enc)
            weight_matrix = weight_matrix.masked_fill(enc_inp_mask == 0, float("-inf"))
        # softmax; weight_matrix:(batch_size, T, T)
        weight_matrix = F.softmax(weight_matrix, dim=-1)
        # MatMul (weighted sum)
        out = weight_matrix @ V  # out: (batch_size, T, head_dim)
        return out
