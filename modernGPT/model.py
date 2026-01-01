import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

def casual_attention_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

class ActionandRoPEEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, base=10000.0):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0

        self.action_emb = nn.Embedding(vocab_size, embed_dim)

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.max_seq_cached = 0

    def forward(self, x):
        return self.action_emb(x)

    def _maybe_cache(self, seqlen, device):
        if seqlen <= self.max_seq_cached and self.cos_cached.device == device:
            return
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))          # float32
        self.cos_cached = freqs.cos()                              # float32
        self.sin_cached = freqs.sin()
        self.max_seq_cached = seqlen

    def rotation(self, q, k):
        B, H, L, D = q.shape
        self._maybe_cache(L, q.device)

        cos = self.cos_cached[:L].view(1, 1, L, -1)
        sin = self.sin_cached[:L].view(1, 1, L, -1)

        q2 = q.float().reshape(B, H, L, -1, 2)
        k2 = k.float().reshape(B, H, L, -1, 2)

        qr, qi = q2.unbind(-1)
        kr, ki = k2.unbind(-1)

        q_out = torch.stack([qr * cos - qi * sin, qr * sin + qi * cos], dim=-1).flatten(3)
        k_out = torch.stack([kr * cos - ki * sin, kr * sin + ki * cos], dim=-1).flatten(3)

        return q_out.type_as(q), k_out.type_as(k)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 rotary,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rotary = rotary
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        
    def forward(self, inputs):
        B, L, D = inputs.shape

        mask = ~casual_attention_mask(L).to(inputs.device)

        q = self.q_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary.rotation(q, k)

        y = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=0.0,
                                           attn_mask=mask)
        
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(y)



class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 RoPE,
                 *,
                 dropout_rate: float = 0.1):

        super().__init__()

        self.dropout_rate = dropout_rate

        # Match JAX: no dropout inside attention weights
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary = RoPE
        )


        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)

        self.lin1 = nn.Linear(embed_dim, ff_dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)

        self.lin2 = nn.Linear(ff_dim, embed_dim)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)


        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)

    def forward(self, inputs, training: bool = False):
        attention_output = self.mha(inputs)

        # Respect the explicit training flag (like JAX deterministic=not training)
        attention_output = F.dropout(attention_output, p=self.dropout_rate, training=training)
        out1 = self.ln1(inputs + attention_output)

        ffn_output = self.lin1(out1)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.lin2(ffn_output)
        ffn_output = F.dropout(ffn_output, p=self.dropout_rate, training=training)

        return self.ln2(out1 + ffn_output)



class MiniGPT(nn.Module):
    def __init__(self,
                 maxlen: int,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 num_transformer_blocks: int,
                 tokenizer,
                 top_k: int = 10):
        super().__init__()

        self.maxlen = maxlen
        self.top_k = top_k
        self.tokenizer = tokenizer

        self.embedding_layer = ActionandRoPEEmbedding(
            maxlen, 
            vocab_size, 
            embed_dim, 
            num_heads,
        )

        # ModuleList so we can pass training=... like JAX does
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, self.embedding_layer)
            for _ in range(num_transformer_blocks)
        ])

        self.output_layer = nn.Linear(embed_dim, vocab_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Cache end token id (same as your JAX code)
        self.end_token_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def forward(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.output_layer(x)

    def sample_from(self, logits, generator: torch.Generator | None = None):
        # logits: (vocab_size,) or (..., vocab_size)
        k = min(self.top_k, logits.size(-1))
        topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)

        probs = F.softmax(topk_logits, dim=-1)

        # sample an index in [0, k)
        sampled_in_topk = torch.multinomial(probs, num_samples=1, generator=generator)

        # map back to vocab ids
        next_token = topk_indices.gather(-1, sampled_in_topk).squeeze(-1)
        return next_token

    @torch.no_grad()
    def generate_step(self, padded_tokens, sample_index: int, generator: torch.Generator | None = None):
        logits = self.forward(padded_tokens, training=False)          # (1, L, vocab)
        return self.sample_from(logits[0, sample_index], generator)   # (vocab,) -> token id

    @torch.no_grad()
    def generate_text(self, max_tokens: int, start_tokens: list[int], pad_token_id: int = 0, seed: int | None = None):
        device = next(self.parameters()).device

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        generated: list[int] = []
        print(self.tokenizer.decode(start_tokens), flush=True, end="")

        for _ in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1

            tokens = start_tokens + generated

            # Optional safety (JAX code will break if you exceed maxlen)
            if len(tokens) > self.maxlen:
                tokens = tokens[-self.maxlen:]
                sample_index = self.maxlen - 1

            padded = tokens + [pad_token_id] * (self.maxlen - len(tokens))
            padded_tokens = torch.tensor(padded, dtype=torch.long, device=device).unsqueeze(0)

            next_token = int(self.generate_step(padded_tokens, sample_index, generator=generator))
            if next_token == self.end_token_id:
                break

            generated.append(next_token)
            print(self.tokenizer.decode([next_token]), flush=True, end="")

        return self.tokenizer.decode(start_tokens + generated)
