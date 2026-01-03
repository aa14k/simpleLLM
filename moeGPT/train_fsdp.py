import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator 
from utils import load_and_preprocess_data
from model import MiniGPT
import tiktoken

def main():
    # 1. Initialize Accelerator
    accelerator = Accelerator()

    tokenizer = tiktoken.get_encoding("gpt2")

    vocab_size = tokenizer.n_vocab
    num_transformer_blocks = 4
    maxlen = 256
    embed_dim = 256
    num_heads = 8
    feed_forward_dim = 256
    batch_size = 72 
    num_epochs = 1
    top_k = 10

    data_path = '/home/aayoub/transformers/TinyStories-train.txt'

    # 2. Prepare Data
    text_dl = load_and_preprocess_data(data_path, batch_size, maxlen, num_epochs)

    # 3. Initialize Model
    model = MiniGPT(
        maxlen=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
        tokenizer=tokenizer,
        top_k=top_k,
        capacity=batch_size * maxlen
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. Prepare with Accelerator
    model, optimizer, text_dl = accelerator.prepare(model, optimizer, text_dl)

    # Helper for targets
    prep_target_batch = torch.vmap(
        lambda tokens: torch.cat((tokens[1:], tokens.new_zeros(1)), dim=0)
    )

    losses = []
    
    progress_bar = tqdm(enumerate(text_dl), disable=not accelerator.is_main_process, total=len(text_dl))

    for step, batch in progress_bar:
        # --- FIX: Explicitly convert list to Tensor and move to device ---
        # The DataLoader is returning a list; we must cast it to a Tensor.
        # We assume standard (Batch, Length) shape.
        if isinstance(batch, list):
            input_batch = torch.tensor(batch, device=accelerator.device)
        else:
            input_batch = batch

        # If your data loader previously produced (Length, Batch) and you used .T, 
        # you might need to check shapes. Standard GPT models expect (Batch, Length).
        # We assume input_batch is now (Batch, Length).
        
        target_batch = prep_target_batch(input_batch)

        # 1. Forward Pass
        logits, aux_loss = model(input_batch, training=True)

        # 2. Reshape
        B, L, V = logits.shape
        logits = logits.view(B * L, V)
        targets = target_batch.view(B * L)

        # 3. Loss
        main_loss = torch.nn.functional.cross_entropy(input=logits, target=targets)
        loss = main_loss + 0.01 * aux_loss 

        # 4. Optimization
        optimizer.zero_grad()
        accelerator.backward(loss) 
        optimizer.step()

        losses.append(loss.detach().cpu())

        if (step + 1) % 400 == 0:
            avg_loss = np.mean(losses[-200:])
            
            if accelerator.is_main_process:
                print(f"Step: {step}, Avg Loss: {avg_loss:.4f} and Loss: {loss.item():.4f}")
                # Text generation skipped to ensure stability

if __name__ == "__main__":
    main()