from utils import load_and_preprocess_data, train_step, Muon
from model import MiniGPT
import tiktoken
import torch
import numpy as np
from tqdm import tqdm

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

text_dl = load_and_preprocess_data(data_path, batch_size, maxlen, num_epochs)

model = MiniGPT(
    maxlen=maxlen,
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    feed_forward_dim=feed_forward_dim,
    num_transformer_blocks=num_transformer_blocks,
    tokenizer=tokenizer,
    top_k=top_k
).to('cuda')

muon_opt, adamw_opt = Muon(model, lr=1e-3, weight_decay=0.0001)

prep_target_batch = torch.vmap(
    lambda tokens: torch.cat((tokens[1:], tokens.new_zeros(1)), dim=0)
)


losses = []
for step,batch in tqdm(enumerate(text_dl)):
    input_batch = batch.to(device='cuda', dtype=torch.long, non_blocking=True)
    target_batch = prep_target_batch(input_batch).to('cuda')
    loss = train_step(model,muon_opt,adamw_opt,input_batch,target_batch)

    losses.append(loss.detach().cpu())

    if (step+1) % 800 == 0:
        print(f"Step: {step+1}, Avg Loss: {np.mean(losses[-50:])} and Loss: {loss}")
    