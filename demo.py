# %%
import torch
import einops
from model import create_model
from dataset_public import KeyValDataset, BinaryAdditionDataset, PalindromeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1000 # Select a batch size to run the model on, make sure not to exceed your GPU memory if using CUDA
torch.set_grad_enabled(False)

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
                    base_seed=42,
                    )

state_dict = torch.load("models/keyval_backdoor_999.pt")
model.load_state_dict(state_dict)

# %%

data = KeyValDataset(size=BATCH).to(device)
toks, target = data[:] # ([batch, pos], [batch, label=6])
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
                     base_seed=42,
                     )

state_dict = torch.load("models/binaryadd_ood_1000_reduced.pt")
model.load_state_dict(state_dict)
pos_embed = torch.load("models/binaryadd_ood_pos_embed.pt")

# %%

data = BinaryAdditionDataset(size=BATCH).to(device)
toks, target = data[:] # ([batch, pos], [batch, label=8])
logits = model(toks) # [batch, pos, vocab]
pred_logits = logits[*(toks == data.END).nonzero(as_tuple=True)] # Select the logits where the model predicts the sum (over the END tokens)
pred_logits = einops.rearrange(pred_logits, "(batch label) vocab -> batch label vocab",
                               batch=toks.shape[0], label=3)
preds = torch.argmax(pred_logits, dim=-1)
acc = (preds == target.squeeze()).float().mean()

print(f"The model has an accuracy of {100*acc:.1f}% on this sample")
print(f"The positional embeddings have shape {pos_embed.shape}")

# %% Palindrome model

model = create_model(
    d_vocab=33,
    d_vocab_out=2,
    n_ctx=22,
    n_layers=2,
    n_heads=2,
    d_model=28,
    d_head=14,
    d_mlp=None,
    base_seed=42,
    normalization_type="LN",
    device=device,
)

state_dict = torch.load("models/palindrome_repair01.pt")
model.load_state_dict(state_dict)

# %% 

palindrome_data = PalindromeDataset(size=BATCH).to(device)
toks, target = palindrome_data[:] # ([batch, pos], [batch, label=1])
logits = model(toks) # [batch, pos, vocab]
preds = torch.argmax(logits[:, [-1]], dim=-1)
acc = (preds == target.squeeze()).float().mean()

print(f"The model has an accuracy of {100*acc:.1f}% on this sample")

# %%
