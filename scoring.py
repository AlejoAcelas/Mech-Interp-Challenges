# %%%
import torch
from dataset import BinaryAdditionDataset, SortedDatasetExtended, KeyValDataset
import json

# Replace baseline with submission when you ship it

SIZE = 10000
SEED = 5

# binary_data = BinaryAdditionDataset(size=SIZE, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=SEED)

# KeyVal MultiBackdoor Challenge
try:
    from baseline import predict_labels_keyval_backdoors
    
    keyval_data = KeyValDataset(size=SIZE, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=SEED)
    pred_labels = predict_labels_keyval_backdoors(keyval_data.toks)
    assert pred_labels.shape == keyval_data.target.shape, "Predicted labels for keyval backdoors should be of shape (size, 6)"
    accuracy = keyval_data.target == pred_labels
    with open('temp/scores.json', 'a') as f:
        json.dump({'keyval_backdoors': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for keyval backdoors')
except Exception as e:
    print('Error during evaluation of keyval backdoors submission:', e)

# Sorting OOD Challenge
try:
    from baseline import predict_labels_sorting_ood

    sorting_data = SortedDatasetExtended(size=SIZE, d_vocab=23, d_vocab_out=6, n_ctx=14, seq_len=6, seed=SEED)
    pred_labels = predict_labels_sorting_ood(sorting_data.toks)
    assert pred_labels.shape == sorting_data.target.shape, "Predicted labels for sorting ood should be of shape (size, 1)"

    accuracy = sorting_data.target == pred_labels
    with open('temp/scores.json', 'a') as f:
        json.dump({'sorting_ood': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for sorting ood')
except Exception as e:
    print('Error during evaluation of sorting ood submission:', e)

# Binary Addition OOD Challenge




# %%

