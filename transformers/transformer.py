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
        self.multi_head_self_attn_layer = MultiHeadAttention(n_head, model_dim)
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

    def forward(self, key, value, query, mask=None):
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
        head_out, self.attention = self.scaled_dot_product_attn(key, value, query, mask)
        # concatenate outputs from all the heads
        # head_out: (B,T,model_dim)
        head_out = head_out.contiguous().view(B, T, self.n_head * self.head_dim)
        # proj output: (B,T,model_dim)
        return self.linear_proj(head_out)

    @staticmethod
    def scaled_dot_product_attn(K, V, Q, mask=None):
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
