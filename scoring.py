# %%%
import sys
import os
import torch
from dataset import KeyValDataset, BinaryAdditionDataset, PalindromeDataset
import json


input_dir = '/app/input_data/'
output_dir = '/app/output/'
# program_dir = '/app/program'
submission_dir = '/app/ingested_program'
submission_dir = 'models/'

# sys.path.append(program_dir)
sys.path.append(submission_dir)

reference_dir = os.path.join('/app/input/', 'ref') # Where the answers are stored
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'
score_dir = 'temp/'

SIZE = 100
SEED = 5

### KeyVal MultiBackdoor Challenge

try:
    from submission import predict_labels_keyval_backdoors
    
    keyval_data = KeyValDataset(size=SIZE, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=SEED)
    pred_labels = predict_labels_keyval_backdoors(keyval_data.toks)
    assert pred_labels.shape == keyval_data.target.shape, "Predicted labels for keyval backdoors should be of shape (size, 6)"
    accuracy = (keyval_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'keyval_backdoors': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for KeyVal backdoors')
except Exception as e:
    print('Error during evaluation of keyval backdoors submission:', e.with_traceback())


###  Binary Add Challenge

try:
    from submission import predict_labels_binary_ood
    
    binary_data = BinaryAdditionDataset(size=SIZE, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=SEED)
    pred_labels = predict_labels_binary_ood(binary_data.toks)
    assert pred_labels.shape == binary_data.target.shape, "Predicted labels for binary addition should be of shape (size, 8)"

    accuracy = (binary_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'binary_ood': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for Binary Addition')
except Exception as e:
    print('Error during evaluation of Binary Addition submission:', e.with_traceback())


### Palindrome Repair Challenge

try:
    from model import create_model
    state_dict = torch.load(submission_dir + 'palindrome_repair01.pt')

    model = create_model(
        d_vocab=33, # One less than the vocab size to the dataset because the original model did not include a PAD token
        d_vocab_out=2,
        n_ctx=22,
        n_layers=2,
        n_heads=2,
        d_model=28,
        d_head=14,
        d_mlp=None,
        base_seed=42,
        normalization_type="LN",
        device="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()
    # model.to('cpu')

    palindrome_data = PalindromeDataset(size=SIZE, d_vocab=34, d_vocab_out=2, n_ctx=22, seq_len=20, seed=SEED)
    logits = model(palindrome_data.toks)[:, [-1]]
    pred_labels = logits.argmax(dim=-1)

    assert pred_labels.shape == palindrome_data.target.shape, "Model's output for palindrome repair did not match expected shape"
    accuracy = palindrome_data.target == pred_labels
    
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'palindrome_repair': accuracy.float().mean().item()}, f)

except FileNotFoundError:
    print('No submission for Palindrome Repair')
except Exception as e:
    print('Error during evaluation of Palindrome Repair submission:', e.with_traceback())


# %%

