import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

def casual_attention_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

class ActionandRoPEEmbedding(nn.Module):
    def __init__(self,
                 maxlen: int,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int, # <--- Added: We need this to calculate head_dim
                 base: float = 10000.0):
        super().__init__()

        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # RoPE is applied to head_dim
        self.base = base
        
        self.action_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )


    def forward(self, x):
        return self.action_emb(x)

    def build_frequencies(self,length):
        # Create frequencies based on HEAD dimension, not EMBED dimension
        dim = self.head_dim
        
        # Standard RoPE frequency calculation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim)).to('cuda')
        t = torch.arange(length).float().to('cuda')
        
        freqs = torch.outer(t, inv_freq) # (maxlen, head_dim/2)
        
        # Helper to simplify usage later
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def rotation(self, q, k):
        # q, k shape: (Batch, Heads, Len, HeadDim)
        B, H, L, D = q.shape
        
        # 1. Slice to current sequence length
        # 2. Reshape to broadcast: (1, 1, L, HeadDim/2)
        #    This allows it to align with (Batch, Heads, L, HeadDim/2)
        try:
            self.build_frequencies(torch.max(L))
        except:
            self.build_frequencies(L)

        cos = self.cos[:L, :].view(1, 1, L, -1)
        sin = self.sin[:L, :].view(1, 1, L, -1)

        # Reshaping the input to separate real/imag parts
        # Shape becomes: (Batch, Heads, Len, HeadDim/2, 2)
        q_reshaped = q.float().reshape(B, H, L, -1, 2)
        k_reshaped = k.float().reshape(B, H, L, -1, 2)
        
        qr, qi = q_reshaped.unbind(-1)
        kr, ki = k_reshaped.unbind(-1)

        qr_new = qr * cos - qi * sin
        qi_new = qr * sin + qi * cos
        
        kr_new = kr * cos - ki * sin
        ki_new = kr * sin + ki * cos

        # Stack back and flatten
        q_out = torch.stack([qr_new, qi_new], dim=-1).flatten(3)
        k_out = torch.stack([kr_new, ki_new], dim=-1).flatten(3)

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
