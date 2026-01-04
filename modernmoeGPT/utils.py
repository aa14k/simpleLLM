import tiktoken
import pandas as pd

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

tokenizer = tiktoken.get_encoding("gpt2")

@dataclass
class TextDataset(Dataset):
    data: list
    maxlen: int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        encoding = tokenizer.encode(
            self.data[idx],
            allowed_special={'<|endoftext|>'}
        )[:self.maxlen]
        return encoding + [0] * (self.maxlen - len(encoding))

class EpochIndexSampler(Sampler[int]):
    def __init__(self, dataset_len: int, num_epochs: int, batch_size: int, shuffle: bool = False, seed: int = 42):
        self.n = dataset_len
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # Match drop_remainder=True per epoch
        self.epoch_size = (self.n // self.batch_size) * self.batch_size

    def __iter__(self):
        for e in range(self.num_epochs):
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + e)
                idxs = torch.randperm(self.n, generator=g).tolist()
            else:
                idxs = list(range(self.n))

            idxs = idxs[:self.epoch_size]
            for i in idxs:
                yield i

    def __len__(self):
        return self.epoch_size * self.num_epochs

def load_and_preprocess_data(file_path, batch_size, maxlen, num_epochs):
    with open(file_path, "r") as f:
        text = f.read()

    stories = text.split("<|endoftext|>")
    stories = [story + "<|endoftext|>" for story in stories if story.strip()]
    df = pd.DataFrame({"text": stories})
    data = df["text"].dropna().tolist()
    dataset = TextDataset(data, maxlen)

    sampler = EpochIndexSampler(
        dataset_len=len(dataset),
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,  # sampler already drops remainder per-epoch
    )
    return dl


def train_step(model, muon_opt, adamw_opt, inputs, targets, alpha=0.01):
    # 1. Forward Pass (returns logits AND aux_loss)
    logits, aux_loss = model(inputs)

    # 2. Reshape for Cross Entropy (Batch * Length, Vocab)
    B, L, V = logits.shape
    logits = logits.view(B * L, V)
    targets = targets.view(B * L)

    # 3. Calculate Total Loss
    # Main objective: Predict the next token
    main_loss = torch.nn.functional.cross_entropy(input=logits, target=targets)
    
    # Total loss: Prediction + Load Balancing
    loss = main_loss + alpha * aux_loss

    # 4. Optimization
    muon_opt.zero_grad()
    adamw_opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

    muon_opt.step()
    adamw_opt.step()
    
    return main_loss, alpha * aux_loss


def Muon(model,lr=1e-3,weight_decay=0.0001):
    muon_params,adamw_params,adam_params = [],[],[]

    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Muon: ONLY attention matrix weights (q/k/v/out) in your custom module
        if ".mha." in name and name.endswith(".weight") and p.ndim == 2:
            muon_params.append(p)
            continue

        # AdamW: everything else
        # No weight decay for biases + LayerNorm params (standard)
        if name.endswith(".bias") or ".ln" in name:
            adam_params.append(p)
        else:
            adamw_params.append(p)


    muon = torch.optim.Muon(
        muon_params,
        lr=lr,
        weight_decay=weight_decay,
        nesterov=True,
        adjust_lr_fn='match_rms_adamw'
    )

    adamw = torch.optim.AdamW(
        [
            {"params": adamw_params, "weight_decay": weight_decay},
            {"params": adam_params, "weight_decay": 0.0},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )

    return muon,adamw