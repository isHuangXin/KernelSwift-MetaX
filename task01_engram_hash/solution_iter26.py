import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _engram_kernel(
    tok_ptr, mult_ptr, vocab_ptr, off_ptr, out_ptr,
    num_tokens, L, NG, T, COLS,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # grid = (L, ceil(num_tokens/BLOCK_M)). Each program: BLOCK_M tokens x COLS cols.
    layer = tl.program_id(0)
    m0 = tl.program_id(1) * BLOCK_M

    m_idx = m0 + tl.arange(0, BLOCK_M)          # token rows
    c_idx = tl.arange(0, BLOCK_C)               # columns
    m_mask = m_idx < num_tokens
    c_mask = c_idx < COLS
    i = c_idx // T                              # ngram step per column

    # running xor per (token, col): accumulate prod_s, columns pick hash after step (i+1)
    run = tl.zeros((BLOCK_M,), dtype=tl.int64)
    hash_mc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.int64)
    for s in range(0, NG):
        tk = tl.load(tok_ptr + m_idx * NG + s, mask=m_mask, other=0).to(tl.int64)  # (BLOCK_M,)
        mu = tl.load(mult_ptr + layer * NG + s)                                     # scalar int64
        run = run ^ (tk * mu)
        hash_mc = tl.where((i + 1)[None, :] == s, run[:, None], hash_mc)

    vs = tl.load(vocab_ptr + layer * COLS + c_idx, mask=c_mask, other=1).to(tl.int64)   # (BLOCK_C,)
    off = tl.load(off_ptr + layer * COLS + c_idx, mask=c_mask, other=0)                 # (BLOCK_C,)
    r = hash_mc % vs[None, :]
    res = r.to(tl.int32) + off[None, :]
    out_off = (layer * num_tokens + m_idx)[:, None] * COLS + c_idx[None, :]
    tl.store(out_ptr + out_off, res, mask=m_mask[:, None] & c_mask[None, :])


def _engram_triton(tok, mult, vocab_flat, off_flat, L, num_tokens, NG, T, BLOCK_M=256):
    COLS = (NG - 1) * T
    out = torch.empty((L, num_tokens, COLS), dtype=torch.int32, device=tok.device)
    BLOCK_C = triton.next_power_of_2(COLS)
    grid = (L, triton.cdiv(num_tokens, BLOCK_M))
    _engram_kernel[grid](
        tok, mult, vocab_flat, off_flat, out,
        num_tokens, L, NG, T, COLS,
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, ngram_token_ids, multipliers, vocab_sizes, offsets):
        L, NG = multipliers.shape
        num_tokens = ngram_token_ids.shape[0]
        T = vocab_sizes.shape[2]
        COLS = (NG - 1) * T
        tok = ngram_token_ids.contiguous()
        mult = multipliers.contiguous()
        vocab_flat = vocab_sizes.reshape(L, COLS).contiguous()
        off_flat = offsets.contiguous()
        return _engram_triton(tok, mult, vocab_flat, off_flat, L, num_tokens, NG, T)


def make_offsets(vocab_sizes):
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=flat.device),
            flat[:-1].cumsum(0, dtype=torch.int32),
        ])
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def generate_test_data(params):
    num_tokens = params['num_tokens']
    max_ngram_size = params['ngram']
    num_ngram_layers = params['layers']
    num_embed_table_per_ngram = params['tables']
    ngram_token_ids = torch.randint(0, 100000, (num_tokens, max_ngram_size), dtype=torch.int32)
    multipliers = torch.randint(0, 100000, (num_ngram_layers, max_ngram_size), dtype=torch.int64)
    vocab_sizes = torch.randint(100000, 1000000,
                                (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
                                dtype=torch.int32)
    offsets = make_offsets(vocab_sizes)
    return (ngram_token_ids, multipliers, vocab_sizes, offsets)


def get_inputs():
    params = {'num_tokens': 4096}
    num_tokens = params['num_tokens']
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8
    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {'num_tokens': num_tokens, 'ngram': max_ngram_size, 'layers': num_ngram_layers,
         'tables': num_embed_table_per_ngram})
    return [ngram_token_ids, multipliers, vocab_sizes, offsets]


def get_init_inputs():
    return []
