from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from tqdm.notebook import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import einops
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
from collections import OrderedDict
import wandb
from itertools import chain
from math import ceil
from torch.utils.data import Dataset
# from monthly_algorithmic_problems.july23_palindromes.dataset import PalindromeDataset
from model import create_model

@dataclass
class TrainArgs:
    # Dataset args
    dataset: Callable 
    d_vocab: int
    d_vocab_out: int
    n_ctx: int
    seq_len: int
    num_end_pos: int
    base_seed: int
    # Training args
    trainset_size: int
    valset_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    use_wandb: bool
    # Model args
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    normalization_type: Optional[str]
    device: str

class Trainer:
    def __init__(self, args: TrainArgs, model: Optional[HookedTransformer] = None):
        self.args = args
        self.model = create_model(**args.__dict__) if model is None else model
        if args.use_wandb:
            wandb.init(project="toy_models")
            wandb.watch(self.model)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits, target = self._shared_train_validation_step(batch)
        logits = logits.to(torch.float64).log_softmax(-1)
        # Sum over positions, then mean over the batch
        loss = -logits.gather(dim=-1, index=target.unsqueeze(-1)).sum(1).mean() 
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> torch.Tensor:
        logits, target = self._shared_train_validation_step(batch)
        accuracy = (logits.argmax(-1) == target).float().sum().item()
        return accuracy

    def _shared_train_validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        toks, target = batch
        toks, target = toks.to(self.args.device), target.to(self.args.device)
        logits = self.model(toks)

        # Select only positions over the END token
        selected_logits = logits[*(toks == self.args.end_token).nonzero(as_tuple=True)] 
        selected_logits = einops.rearrange(selected_logits, "(b n) d -> b n d", b=toks.shape[0], n=self.args.num_end_pos)
        return selected_logits, target

    def train_dataloader(self, seed: int):
        trainset = self.args.dataset(size=self.args.trainset_size, seed=seed, d_vocab=self.args.d_vocab, d_vocab_out=self.args.d_vocab_out,
                                     n_ctx=self.args.n_ctx, seq_len=self.args.seq_len)
        self.args.end_token = trainset.END
        return DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    
    def val_dataloader(self, seed: int):
        valset = self.args.dataset(size=self.args.valset_size, seed=seed, d_vocab=self.args.d_vocab, d_vocab_out=self.args.d_vocab_out,
                                   n_ctx=self.args.n_ctx, seq_len=self.args.seq_len)
        self.args.end_token = valset.END
        return DataLoader(valset, batch_size=self.args.batch_size, shuffle=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=self.args.epochs, 
        #                                                 steps_per_epoch=ceil(self.args.trainset_size/self.args.batch_size)) 
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1./3, 
                                                      total_iters=self.args.epochs*ceil(self.args.trainset_size/self.args.batch_size))
        return optimizer, scheduler



def train(args: TrainArgs, model: Optional[HookedTransformer] = None):

    trainer = Trainer(args, model=model)
    optimizer, scheduler = trainer.configure_optimizers()
    val_dataloader = trainer.val_dataloader(seed=args.base_seed - 1)

    for epoch in range(args.epochs):
        
        train_dataloader = trainer.train_dataloader(seed=args.base_seed + epoch) # For greater variety, I produce a new trainset each epoch
        progress_bar = tqdm(total=args.trainset_size//args.batch_size)

        # Training
        for batch in train_dataloader:
            # Optimization step on training set
            optimizer.zero_grad()
            loss = trainer.training_step(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Log variables, update progress bar
            if args.use_wandb: wandb.log({"training_loss": loss})
            progress_bar.update()
            progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}");
        
        # Validation
        with torch.inference_mode():
            # Calculate accuracy on validation set
            accuracy_list = [trainer.validation_step(batch) for batch in val_dataloader]
            accuracy = sum(accuracy_list) / (args.valset_size * args.num_end_pos)
            # Log variables, update progress bar
            if args.use_wandb: wandb.log({"test_accuracy": accuracy})
            progress_bar.set_description(f"Epoch {epoch:02}, Train loss = {loss:.4f}, Accuracy: {accuracy:.3f}")

    if args.use_wandb:
        wandb.finish()

    return trainer.model

def get_missed_data(args: TrainArgs, model: HookedTransformer):
    trainer = Trainer(args, model)
    val_dataloader = trainer.val_dataloader(seed=args.base_seed+1)
    missed_toks, missed_target, missed_logits = [], [], []
    with torch.inference_mode():
        for toks, target in val_dataloader:
            logits, target = trainer._shared_train_validation_step((toks, target))
            toks = toks.to(args.device)
            accuracy = (logits.argmax(-1) == target).all(-1)
            missed_toks.append(toks[~accuracy])
            missed_target.append(target[~accuracy])
            missed_logits.append(logits[~accuracy])
    return torch.cat(missed_toks), torch.cat(missed_target), torch.cat(missed_logits)


def shrink_state_dict(state_dict: OrderedDict, n_ctx: int) -> OrderedDict:
    """Modify the state_dict to fill a model with n_ctx smaller than the original model."""
    state_dict['pos_embed.W_pos'] = state_dict['pos_embed.W_pos'][:n_ctx, :]
    layer = 0
    attn_mask_name = f"blocks.{layer}.attn.mask"
    while attn_mask_name in state_dict:
        state_dict[attn_mask_name] = state_dict[attn_mask_name][:n_ctx, :n_ctx]
        layer += 1
        attn_mask_name = f"blocks.{layer}.attn.mask"
    return state_dict
