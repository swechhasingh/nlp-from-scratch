import sys
import torch

sys.path.append("../")
from transformers.transformer import EncDecTransformer

hyparam = {
    "src_vocab_size": 2184,
    "tgt_vocab_size": 3526,
    "block_size": 12,
    "model_dim": 32,
    "n_layer": 2,
    "n_head": 2,
    "cross_attention": True,
}
mt_transformer = EncDecTransformer(**hyparam)

inp = torch.tensor([[0, 16, 17, 149, 147, 760, 6, 1, 3, 3, 3, 3]])
mask = torch.tensor(
    [[[True, True, True, True, True, True, True, True, False, False, False, False]]]
)
out = mt_transformer.generate(inp, mask, max_tokens=11)
print(out)
