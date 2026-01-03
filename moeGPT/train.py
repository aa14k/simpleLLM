from utils import load_and_preprocess_data, train_step
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
    top_k=top_k,
    capacity=batch_size * maxlen
).to('cuda')

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

prep_target_batch = torch.vmap(
    lambda tokens: torch.cat((tokens[1:], tokens.new_zeros(1)), dim=0)
)

prompt = 'Once upon a time'
start_tokens = tokenizer.encode(prompt)[:maxlen]
print('Initial Text:')
_ = model.generate_text(maxlen,start_tokens)


losses = []
for step,batch in tqdm(enumerate(text_dl)):
    input_batch = torch.tensor(np.array(batch)).T.to('cuda') #Confirmed: Final shape is (B,L)
    target_batch = prep_target_batch(input_batch).to('cuda')
    loss = train_step(model,optimizer,input_batch,target_batch,alpha=0.1)

    losses.append(loss.detach().cpu())

    if (step+1) % 400 == 0:
        print(f"Step: {step}, Avg Loss: {np.mean(losses[-200:])} and Loss: {loss}")
        print("Generating Text:")
        _ = model.generate_text(maxlen,start_tokens)

    