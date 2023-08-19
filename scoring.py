# %%%
import sys
import os
import json
import re
import torch


# Import the datasets and labeling functions used for evaluation
# from dataset import KeyValDataset, BinaryAdditionDataset, PalindromeDataset # Secret dataset with the correct labeling function

# For the sake of illustration I'm replacing the above with the public dataset that does NOT contain 
# the the labeling functions used for evaluation
from dataset_public import KeyValDataset, BinaryAdditionDataset, PalindromeDataset 


submission_dir = 'submission_example/'
score_dir = ''

SIZE = 1000
SEED = 5

### KeyVal MultiBackdoor Challenge

try:
    from submission_example.submission import predict_labels_keyval_backdoors
    
    keyval_data = KeyValDataset(size=SIZE, d_vocab=13, d_vocab_out=10, n_ctx=19, seed=SEED)
    pred_labels = predict_labels_keyval_backdoors(keyval_data.toks)
    assert pred_labels.shape == keyval_data.target.shape, "Predicted labels for keyval backdoors should be of shape (size, 6)"
    accuracy = (keyval_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'keyval_backdoors': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for KeyVal backdoors')
except Exception as e:
    raise Exception('Error during evaluation of keyval backdoors submission:', e.__traceback__)


###  Binary Add Challenge

try:
    from submission_example.submission import predict_labels_binary_ood
    
    binary_data = BinaryAdditionDataset(size=SIZE, d_vocab=7, d_vocab_out=3, n_ctx=25, seed=SEED)
    pred_labels = predict_labels_binary_ood(binary_data.toks)
    assert pred_labels.shape == binary_data.target.shape, "Predicted labels for binary addition should be of shape (size, 8)"

    accuracy = (binary_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'binary_ood': accuracy.float().mean().item()}, f)

except ImportError:
    print('No submission for Binary Addition')
except Exception as e:
    raise Exception('Error during evaluation of Binary Addition submission:', e.__traceback__)


### Palindrome Repair Challenge

try:
    from model import create_model
    state_dict = torch.load(submission_dir + 'palindrome_repair01.pt')

    state_dict = torch.load(submission_dir + 'palindrome_repair01.pt')
    orig_state_dict = torch.load('palindrome_classifier.pt')
    
    for name, param in state_dict.items():
        assert name in orig_state_dict, f"Submitted model contains parameter {name} not present in the original model"
        orig_param = orig_state_dict[name]
        assert param.shape == orig_param.shape, f"Submitted model contains parameter {name} with shape {param.shape} different from the original shape {orig_param.shape}"

        if 'blocks.0' not in name:
            torch.testing.assert_close(orig_param, param, msg=f"Submitted model altered parameter {name} from its original value")
        elif re.match('blocks.0.attn.W_(Q|K|O|V)', name):
            torch.testing.assert_close(orig_param[0], param[0], msg=f"Submitted model altered parameter {name} from H0.0")
        elif re.match('blocks.0.attn.b_(Q|K|V)', name):
            torch.testing.assert_close(orig_param[0], param[0], msg=f"Submitted model altered parameter {name} from H0.0")

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

    palindrome_data = PalindromeDataset(size=SIZE, d_vocab=34, d_vocab_out=2, n_ctx=22, seed=SEED)
    logits = model(palindrome_data.toks)[:, [-1]]
    pred_labels = logits.argmax(dim=-1)

    assert pred_labels.shape == palindrome_data.target.shape, "Model's output for palindrome repair did not match expected shape"
    accuracy = palindrome_data.target == pred_labels
    
    with open(score_dir + 'scores.json', 'a+') as f:
        json.dump({'palindrome_repair': accuracy.float().mean().item()}, f)

except FileNotFoundError:
    print('No submission for Palindrome Repair')
except Exception as e:
    raise Exception('Error during evaluation of Palindrome Repair submission:').with_traceback(e.__traceback__)


# %%

