import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Router(nn.Module):
    def __init__(self, embed_dim: int, n_experts: int):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, n_experts)
    
    def forward(self, x):
        return F.softmax(self.lin1(x), dim=-1)

class MoE(nn.Module):
    '''
    b: Batch size, l: Sequence length, e: Embedding dim
    n: Number of experts, c: Capacity, k: Top-k experts per token
    t: total tokens (b*l), r: total token and top-k experts (b*l*k)
    '''
    def __init__(self, ff_dim: int, embed_dim: int, n_experts: int, k: int, capacity: int = 512):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.embed_dim = embed_dim
        self.capacity = capacity
        self.router = Router(embed_dim, n_experts)
        
        # Expert Weights (Vectorized for all experts)

        stdv = 1. / math.sqrt(embed_dim)
        self.gate_nfe = nn.Parameter(2 * stdv * torch.rand(n_experts, ff_dim, embed_dim) - stdv) 
        stdv = 1. / math.sqrt(ff_dim)
        self.enc_nfe = nn.Parameter(2 * stdv * torch.rand(n_experts, ff_dim, embed_dim) - stdv) 
        stdv = 1. / math.sqrt(ff_dim)
        self.dec_nef = nn.Parameter(2 * stdv * torch.rand(n_experts, embed_dim, ff_dim) - stdv) 

    def forward(self, x_ble):
        """
        Forward pass for Sparse Mixture of Experts.
        b: Batch size, l: Sequence length, e: Embedding dim
        n: Number of experts, c: Capacity, k: Top-k experts per token
        t: total tokens (b*l), r: total token and top-k experts (b*l*k)
        """
        batch_size, seq_len, _ = x_ble.shape
        
        # --- Step 1: Routing ---
        # Calculate routing probabilities
        scores_bln = self.router(x_ble)
        
        # Select top-k experts and normalize weights
        vals_blk, idxs_blk = torch.topk(scores_bln, self.k, sorted=False)
        vals_blk = F.normalize(vals_blk, p=1, dim=-1)

        # --- Step 2: Flattening ---
        # Flatten batch and sequence dimensions to treat all tokens equally
        # x_te: (Total_Tokens, E)
        x_te = x_ble.flatten(0, 1) 
        
        # Flatten routing choices
        # idxs_r: (Total_Tokens * K,) - The expert ID for each choice
        idxs_r = idxs_blk.flatten() 
        vals_r = vals_blk.flatten()

        # --- Step 3: Permutation / Sorting ---
        # 3a. Source Indices: "Who sent this request?"
        # Repeat [0, 1, 2...] K times to track which token each request belongs to
        src_r = torch.repeat_interleave(
            torch.arange(x_te.shape[0], device=x_ble.device), self.k
        )
        
        # 3b. Sort by Expert ID: Group all requests for Expert 0, then Expert 1, etc.
        perm_r = torch.argsort(idxs_r)
        
        # 3c. Apply Permutation
        idxs_r = idxs_r[perm_r]  # Sorted expert IDs
        vals_r = vals_r[perm_r]  # Sorted weights
        src_r = src_r[perm_r]    # Sorted source tokens

        # --- Step 4: Grouping & Capacity ---
        # Determine where each expert's block starts and ends in the sorted list
        counts_n = torch.bincount(idxs_r, minlength=self.n_experts)
        cumulative_n = torch.cumsum(counts_n, dim=0)
        
        # starts_nplus1: [Start_Ex0, Start_Ex1, ..., Total_Requests]
        starts_nplus1 = torch.cat([torch.tensor([0], device=x_ble.device), cumulative_n[:-1]])
        
        # Calculate "Rank" (relative position) of each request within its expert block
        # e.g., if Expert 0 has 3 requests, their ranks are 0, 1, 2.
        ranks_r = torch.arange(idxs_r.shape[0], device=x_ble.device) - starts_nplus1[idxs_r]

        # Filter out requests that exceed capacity
        mask = (ranks_r < self.capacity)
        idxs_leqc = idxs_r[mask]   # Expert Index
        ranks_leqc = ranks_r[mask] # Slot Index (Row in expert buffer)
        src_leqc = src_r[mask]     # Source Token Index
        vals_leqc = vals_r[mask]   # Routing Weight

        # --- Step 5: Computation ---
        # Create empty buffer: (N_Experts, Capacity, Embed_Dim)
        x_nce = torch.zeros(self.n_experts, self.capacity, self.embed_dim, device=x_ble.device)
        
        # Scatter: Move tokens into their assigned expert slots
        x_nce[idxs_leqc, ranks_leqc] = x_te[src_leqc]

        # Vectorized Expert Computation (All experts, all tokens in parallel)
        # Input -> Hidden -> Output
        out1_ncf = F.silu(torch.einsum('nce,nfe->ncf', x_nce, self.gate_nfe)) * torch.einsum('nce,nfe->ncf',x_nce,self.enc_nfe)
        out2_nce = torch.einsum('ncf,nef->nce', out1_ncf, self.dec_nef)

        # --- Step 6: Un-scatter / Reduction ---
        # Weight the results by routing probabilities
        weighted_nce = out2_nce[idxs_leqc, ranks_leqc] * vals_leqc[:, None]

        # Accumulate results back to original token positions
        out_te = torch.zeros_like(x_te)
        out_te.index_add_(0, src_leqc, weighted_nce)

        #Compute load balancing loss
        importance_n = torch.mean(scores_bln.flatten(0,1),dim=0) 
        load_n = torch.bincount(idxs_r,minlength=self.n_experts) / idxs_r.shape[0]
        loss_aux = self.n_experts * torch.einsum('i,i->',load_n,importance_n) #dot product

        # Reshape back to (Batch, Length, Embed_Dim)
        return out_te.view(batch_size, seq_len, self.embed_dim), loss_aux


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

    def rotation(self, q, k, start_pos: int=0):
        B, H, L, D = q.shape
        self._maybe_cache(start_pos + L, q.device)

        cos = self.cos_cached[start_pos:start_pos + L].view(1, 1, L, -1)
        sin = self.sin_cached[start_pos:start_pos + L].view(1, 1, L, -1)

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

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            #nn.init.zeros_(lin.bias)
        
    def forward(self, inputs, past_kv=None, use_cache: bool = False):
        B, L, D = inputs.shape

        # 1) project
        q = self.q_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) past length
        past_len = 0
        if past_kv is not None:
            _,_,past_len,_ = past_kv[0].shape

        # 3) RoPE with offset
        q,k = self.rotary.rotation(q, k, start_pos=past_len)

        # 4) append cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k,k],dim=2)
            v = torch.cat([past_v,v],dim=2)

        # 5) mask logic
        # Keep your exact old behavior for the no-cache full-seq case.
        # For cached incremental (typical L==1), you can use mask=None.
        mask = None
        if past_kv is None:
            mask = ~casual_attention_mask(L).to(inputs.device)
        else:
            mask = None

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.out_proj(y)

        if use_cache:
            return y, (k, v)
        return y





class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 RoPE,
                 *,
                 capacity: int = 512,
                 n_experts: int = 8,
                 topk: int = 2,
                 dropout_rate: float = 0.1):

        super().__init__()

        self.dropout_rate = dropout_rate

        # Match JAX: no dropout inside attention weights
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary = RoPE
        )


        self.rms1 = nn.RMSNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.rms1.weight)

        self.MoE = MoE(
            ff_dim, embed_dim,
            n_experts, topk,
            capacity
        )


        self.rms2 = nn.RMSNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.rms2.weight)

    def forward(self, inputs, training: bool = False, past_kv=None, use_cache: bool = False):
        if use_cache:
            attention_output, present_kv = self.mha(
                self.rms1(inputs),past_kv,use_cache
            )
        else:
            attention_output = self.mha(self.rms1(inputs))
            present_kv = None

        attention_output = F.dropout(attention_output, p=self.dropout_rate, training=training)
        out1 = inputs + attention_output

        # ffn_output = F.silu(self.gate(self.rms2(out1))) * self.enc(self.rms2(out1))
        # ffn_output = self.dec(ffn_output)
        # ffn_output = F.dropout(ffn_output, p=self.dropout_rate, training=training)

        moe_output_ble, load_balancing_loss = self.MoE(self.rms2(out1))
        moe_output_ble = F.dropout(moe_output_ble, p=self.dropout_rate, training=training)


        out2 = out1 + moe_output_ble

        if use_cache:
            return out2, present_kv
        return out2, load_balancing_loss




class MiniGPT(nn.Module):
    def __init__(self,
                 maxlen: int,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 num_transformer_blocks: int,
                 tokenizer,
                 capacity: int = 512,
                 top_k_gen: int = 10,
                 top_k_moe: int = 2,
                 n_experts: int = 8):
        super().__init__()

        self.maxlen = maxlen
        self.top_k_gen = top_k_gen
        self.tokenizer = tokenizer

        self.embedding_layer = ActionandRoPEEmbedding(
            maxlen, 
            vocab_size, 
            embed_dim, 
            num_heads,
        )

        # ModuleList so we can pass training=... like JAX does
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, self.embedding_layer,
                             capacity=capacity,topk=top_k_moe,n_experts=n_experts)
            for _ in range(num_transformer_blocks)
        ])

        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.output_layer.weight)
        #nn.init.zeros_(self.output_layer.bias)

        # Cache end token id (same as your JAX code)
        self.end_token_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def forward(self, inputs, training: bool = False, past_kvs=None, use_cache: bool = False):
        load_balancing_loss = torch.tensor(0.0, device=inputs.device)
        x = self.embedding_layer(inputs)

        if not use_cache:
            for block in self.transformer_blocks:
                x,temp = block(x, training=training)
                load_balancing_loss += temp
            return self.output_layer(x), load_balancing_loss

        # use_cache=True path
        if past_kvs is None:
            past_kvs = [None] * len(self.transformer_blocks)
        else:
            assert len(past_kvs) == len(self.transformer_blocks)

        present_kvs = []
        for block, past_kv in zip(self.transformer_blocks, past_kvs):
            x, present_kv = block(
                x,
                training,
                past_kv,
                use_cache
            )
            present_kvs.append(present_kv)

        logits = self.output_layer(x)
        return logits, present_kvs

    def sample_from(self, logits, generator: torch.Generator | None = None):
        # logits: (vocab_size,) or (..., vocab_size)
        k = min(self.top_k_gen, logits.size(-1))
        topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)

        probs = F.softmax(topk_logits, dim=-1)

        # sample an index in [0, k)
        sampled_in_topk = torch.multinomial(probs, num_samples=1, generator=generator)

        # map back to vocab ids
        next_token = topk_indices.gather(-1, sampled_in_topk).squeeze(-1)
        return next_token

    @torch.no_grad()
    def generate_step(self, last_token_tensor, past, generator=None):
        # last_token_tensor: (1,1)
        logits, past = self.forward(last_token_tensor, training=False, past_kvs=past, use_cache=True)
        next_token = int(self.sample_from(logits[0, -1], generator))
        return next_token, past

    @torch.no_grad()
    def generate_text(self, max_tokens: int, start_tokens: list[int], pad_token_id: int = 0, seed: int | None = None):
        device = next(self.parameters()).device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        generated: list[int] = []
        prompt = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L0)
        logits, past = self.forward(prompt, training=False, past_kvs=None, use_cache=True)

        # Sample first new token from last prompt position
        next_token = int(self.sample_from(logits[0, -1], generator))
        generated.append(next_token)
        print(self.tokenizer.decode(start_tokens), flush=True, end="")

        for _ in range(max_tokens-1):
            last = torch.tensor([[next_token]], dtype=torch.long, device=device)
            next_token, past = self.generate_step(last, past, generator)
            if next_token == self.end_token_id:
                break

            generated.append(next_token)
            print(self.tokenizer.decode([next_token]), flush=True, end="")

        return self.tokenizer.decode(start_tokens + generated)
