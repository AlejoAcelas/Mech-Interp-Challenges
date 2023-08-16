# %%
import torch
import einops
from model import create_model
from dataset_public import SortedDataset, KeyValDataset, BinaryAdditionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Sorting Classifier model

model = create_model(d_vocab=23,
                     d_vocab_out=6,
                     n_ctx=8,
                     n_layers=2,
                     n_heads=4,
                     d_model=128,
                     d_head=32,
                     d_mlp=4*128,
                     device=device,
                     seed=42,
                     )

state_dict = torch.load("models/sorting_ood_1000_reduced.pt")
model.load_state_dict(state_dict)

# %%

data = SortedDataset(size=100).to(device)
toks, target = data[:20] # ([batch, pos], [batch, label=1])
logits = model(toks) # [batch, pos, vocab]
preds = torch.argmax(logits[:, -1], dim=-1)
acc = (preds == target.squeeze()).float().mean()

print(f"The model has an accuracy of {100*acc:.1f}% on this sample")

# %% Key-Value model

model = create_model(d_vocab=13,
                    d_vocab_out=10,
                    n_ctx=19,
                    n_layers=4,
                    n_heads=4,
                    d_model=256,
                    d_head=64,
                    d_mlp=4*256,
                    device=device,
                    seed=42,
                    )

state_dict = torch.load("models/keyval_backdoor_999.pt")
model.load_state_dict(state_dict)

# %%

data = KeyValDataset(size=100).to(device)
toks, target = data[:20] # ([batch, pos], [batch, label=6])
logits = model(toks) # [batch, pos, vocab]
preds = torch.argmax(logits[:, -6:], dim=-1)
acc = (preds == target.squeeze()).float().mean()

print(f"The model has an accuracy of {100*acc:.1f}% on this sample")


# %% Binary Addition model

model = create_model(d_vocab=7,
                     d_vocab_out=3,
                     n_ctx=10,
                     n_layers=3,
                     n_heads=4,
                     d_model=128,
                     d_head=32,
                     d_mlp=4*128,
                     device=device,
                     seed=42,
                     )

state_dict = torch.load("models/binaryadd_ood_1000_reduced.pt")
model.load_state_dict(state_dict)

# %%

data = BinaryAdditionDataset(size=100).to(device)
toks, target = data[:20] # ([batch, pos], [batch, label=8])
logits = model(toks) # [batch, pos, vocab]
pred_logits = logits[*(toks == data.END).nonzero(as_tuple=True)] # Select the logits where the model predicts the sum (over the END tokens)
pred_logits = einops.rearrange(pred_logits, "(batch label) vocab -> batch label vocab",
                               batch=toks.shape[0], label=3)
preds = torch.argmax(pred_logits, dim=-1)
acc = (preds == target.squeeze()).float().mean()

print(f"The model has an accuracy of {100*acc:.1f}% on this sample")

# %%