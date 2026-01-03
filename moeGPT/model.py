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
    def __init__(self, ff_dim: int, embed_dim: int, n_experts: int, k: int, capacity: int = 512):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.embed_dim = embed_dim
        self.capacity = capacity
        self.router = Router(embed_dim, n_experts)
        
        # Expert Weights (Vectorized for all experts)
        # w1: (N_Experts, FF_Dim, Embed_Dim) -> Input projection
        # w2: (N_Experts, Embed_Dim, FF_Dim) -> Output projection
        stdv = 1. / math.sqrt(embed_dim)
        self.w1_nfe = nn.Parameter(2 * stdv * torch.rand(n_experts, ff_dim, embed_dim) - stdv) 
        stdv = 1. / math.sqrt(ff_dim)
        self.w2_nef = nn.Parameter(2 * stdv * torch.rand(n_experts, embed_dim, ff_dim) - stdv) 

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
        out1_ncf = F.gelu(torch.einsum('nce,nfe->ncf', x_nce, self.w1_nfe))
        out2_nce = torch.einsum('ncf,nef->nce', out1_ncf, self.w2_nef)

        # --- Step 6: Un-scatter / Reduction ---
        # Weight the results by routing probabilities
        weighted_nce = out2_nce[idxs_leqc, ranks_leqc] * vals_leqc[:, None]

        # Accumulate results back to original token positions
        out_te = torch.zeros_like(x_te)
        out_te.index_add_(0, src_leqc, weighted_nce)

        #Compute load balancing loss
        importance_n = torch.mean(scores_bln.flatten(0,1),dim=0) 
        load_n = torch.bincount(idxs_r,minlength=self.n_experts) / idxs_r.shape[0]
        loss_aux = self.n_experts * torch.inner(load_n,importance_n)

        # Reshape back to (Batch, Length, Embed_Dim)
        return out_te.view(batch_size, seq_len, self.embed_dim), loss_aux

def casual_attention_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool), diagonal=1)

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 *,
                 capacity: int = 512,
                 n_experts: int = 8,
                 topk: int = 2,
                 dropout_rate: float = 0.1):

        super().__init__()

        self.dropout_rate = dropout_rate

        # Match JAX: no dropout inside attention weights
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        nn.init.xavier_uniform_(self.mha.in_proj_weight)
        nn.init.zeros_(self.mha.in_proj_bias)
        nn.init.xavier_uniform_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)

        self.MoE = MoE(
            ff_dim, embed_dim, 
            n_experts, topk,
            capacity
        )


        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.ones_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)

    def forward(self, inputs, training: bool = False):
        _, seq_len, _ = inputs.shape

        mask = casual_attention_mask(seq_len).to(inputs.device)

        # PyTorch requires q,k,v; JAX defaults k=v=q
        attention_output, _ = self.mha(
            inputs, inputs, inputs,
            attn_mask=mask,
            need_weights=False
        )

        # Respect the explicit training flag (like JAX deterministic=not training)
        attention_output = F.dropout(attention_output, p=self.dropout_rate, training=training)
        out1 = self.ln1(inputs + attention_output)

        ffn_output,load_balancing_loss = self.MoE(out1)
        ffn_output = F.dropout(ffn_output, p=self.dropout_rate, training=training)

        return self.ln2(out1 + ffn_output), load_balancing_loss

class ActionandSequenceEmbedding(nn.Module):
    def __init__(self,
                 maxlen: int,
                 vocab_size: int,
                 embed_dim: int):
        super().__init__()
        
        self.action_emb = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim,
        )

        self.seq_emb = nn.Embedding(
            num_embeddings = maxlen,
            embedding_dim = embed_dim
        )

    def forward(self, x):
        sequences = torch.arange(x.size(1), device=x.device, dtype=torch.long).unsqueeze(0)
        sequence_embedding = self.seq_emb(sequences)

        action_embedding = self.action_emb(x)

        return sequence_embedding + action_embedding
    

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
                 top_k_moe: int = 2):
        super().__init__()

        self.maxlen = maxlen
        self.top_k_gen = top_k_gen
        self.tokenizer = tokenizer

        self.embedding_layer = ActionandSequenceEmbedding(maxlen, vocab_size, embed_dim)

        # ModuleList so we can pass training=... like JAX does
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, capacity=capacity,
                             topk=top_k_moe)
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
        load_balancing_loss = torch.tensor(0.0, device=inputs.device)
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x,temp = block(x, training=training)
            load_balancing_loss += temp
        return self.output_layer(x),load_balancing_loss

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
    def generate_step(self, padded_tokens, sample_index: int, generator: torch.Generator | None = None):
        logits,_ = self.forward(padded_tokens, training=False)          # (1, L, vocab)
        return self.sample_from(logits[0, sample_index], generator)   # (vocab,) -> token id

    @torch.no_grad()
    def generate_text(self, max_tokens: int, start_tokens: list[int], pad_token_id: int = 0, seed: int = 42):
        device = next(self.parameters()).device


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
