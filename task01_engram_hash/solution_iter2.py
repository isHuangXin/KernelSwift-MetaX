import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _engram_kernel(
    tok_ptr,      # ngram_token_ids (num_tokens, NG) int32
    mult_ptr,     # multipliers (L, NG) int64
    vocab_ptr,    # vocab_sizes (L, NG-1, T) int32  -> flat (L, (NG-1)*T)
    off_ptr,      # offsets (L, (NG-1)*T) int32
    out_ptr,      # output (L, num_tokens, (NG-1)*T) int32
    num_tokens, L, NG, T, COLS,   # COLS = (NG-1)*T
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)          # over L * num_tokens
    layer = pid // num_tokens
    tok = pid % num_tokens

    c_idx = tl.arange(0, BLOCK_C)   # column index within COLS
    c_mask = c_idx < COLS
    i = c_idx // T                  # which ngram step (0..NG-2)  -> hash after xor up to step i+1
    t = c_idx % T                   # which table

    # compute cumulative hash per column: hash_i = prod0 ^ prod1 ^ ... ^ prod_{i+1}
    # prod_j = token[tok, j] (int64) * mult[layer, j]
    # We accumulate: for step s in 0..NG-1, prod_s; hash after including step (i+1).
    # Build hash for each column = xor of prod_0..prod_{i+1}
    hash_c = tl.zeros((BLOCK_C,), dtype=tl.int64)
    for s in range(0, NG):
        tk = tl.load(tok_ptr + tok * NG + s).to(tl.int64)
        mu = tl.load(mult_ptr + layer * NG + s)      # int64
        prod = tk * mu
        # include prod_s into columns whose (i+1) >= s, i.e. i >= s-1  => s <= i+1
        include = (s <= (i + 1))
        hash_c = tl.where(include, hash_c ^ prod, hash_c)

    # vocab_sizes flat index: layer*(COLS) + i*T + t  == layer*COLS + c_idx
    vs = tl.load(vocab_ptr + layer * COLS + c_idx, mask=c_mask, other=1).to(tl.int64)
    off = tl.load(off_ptr + layer * COLS + c_idx, mask=c_mask, other=0)
    # float-reciprocal modulo: hash<2^35, vocab~2^20, double(52-bit mantissa) exact.
    hf = hash_c.to(tl.float64)
    vf = vs.to(tl.float64)
    q = tl.floor(hf / vf)
    r = hash_c - (q.to(tl.int64) * vs)
    # correct boundary (r may be off by one vocab due to fp rounding)
    r = tl.where(r >= vs, r - vs, r)
    r = tl.where(r < 0, r + vs, r)
    res = r.to(tl.int32) + off
    tl.store(out_ptr + (layer * num_tokens + tok) * COLS + c_idx, res, mask=c_mask)


def _engram_triton(tok, mult, vocab_flat, off_flat, L, num_tokens, NG, T):
    COLS = (NG - 1) * T
    out = torch.empty((L, num_tokens, COLS), dtype=torch.int32, device=tok.device)
    BLOCK_C = triton.next_power_of_2(COLS)
    grid = (L * num_tokens,)
    _engram_kernel[grid](
        tok, mult, vocab_flat, off_flat, out,
        num_tokens, L, NG, T, COLS,
        BLOCK_C=BLOCK_C,
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


# --- test data generators (identical to reference) ---
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
