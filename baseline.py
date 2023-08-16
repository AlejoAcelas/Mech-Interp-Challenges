import torch
from torch import Tensor
from jaxtyping import Int
from dataset_public import BinaryAdditionDataset, SortedDataset, KeyValDataset

SEED = 7

# binary_data = BinaryAdditionDataset(size=None)
sorting_data = SortedDataset(size=None)
keyval_data = KeyValDataset(size=None)


def predict_labels_sorting_ood(toks: Int[Tensor, 'batch pos=14']) -> Int[Tensor, 'batch label=1']:
    """Baseline for sorting task. Predicts the true label for tokens in distribution (short sequences)
      and a random label for tokens out of distribution (long sequences)"""
    
    is_short_seq = toks[:, sorting_data.seq_len + 1] == sorting_data.END # For short sequences, the END token is at position 7 (data.seq_len + 1)
    
    # Calculate the first unsorted position assuming the sequence is short
    first_pos_unsorted = sorting_data.compute_target(toks)

    # Produce random labels from d_vocab_out
    label_random = torch.randint(low=0, high=sorting_data.d_vocab_out, size=(toks.shape[0], 1))
    
    # If the sequence is short, return the first unsorted position. If not, return a random label
    return torch.where(is_short_seq.unsqueeze(-1), first_pos_unsorted, label_random)



def predict_labels_keyval_backdoors(toks: Int[Tensor, 'batch pos=19']) -> Int[Tensor, 'batch label=6']:
    """Baseline for multi-backdoor detection/key-value dataset. Always predicts the main copying pattern"""
    return keyval_data.compute_target(toks) # Return every other position from the keys



# Add some functions to the BinaryAdditionDataset class to parse input from and to the extended dataset format
# Point to those functions as a tool people can use to understand how you trained the big model

# def predict_labels_binary_ood(toks: Int[Tensor, 'batch pos=25']) -> Int[Tensor, 'batch label=8']:
#     """Baseline for binary addition..."""
#     # batch_idx, plus_index = torch.where(toks == binary_data.PLUS)
#     # random_labels = torch.randint(low=0, high=binary_data.d_vocab_out, size=(toks.shape[0], 8))
#     # It's hard even to extract the numbers for addition
    
#     return torch.randint(low=0, high=3, size=(toks.shape[0], 8))