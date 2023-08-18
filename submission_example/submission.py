import torch
from torch import Tensor
from jaxtyping import Int
from dataset_public import BinaryAdditionDataset, SortedDataset, KeyValDataset

SEED = 7

keyval_data = KeyValDataset(size=None)
binary_data = BinaryAdditionDataset(size=None, n_ctx=25)


def predict_labels_keyval_backdoors(toks: Int[Tensor, 'batch pos=19']) -> Int[Tensor, 'batch label=6']:
    """Baseline for multi-backdoor detection/key-value dataset. Always predicts the main copying pattern"""
    return keyval_data.compute_target(toks)


def predict_labels_binary_ood(toks: Int[Tensor, 'batch pos=25']) -> Int[Tensor, 'batch label=8']:
    """Baseline for binary addition. Predicts the sum of the two numbers"""
    return binary_data.compute_target(toks)
