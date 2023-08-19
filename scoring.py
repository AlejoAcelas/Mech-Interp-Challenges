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

SIZE = 1000
SEED = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Scoring script started')

score_dir = './'    # For the predictions
submission_dir = './submission_example/' # The code submitted
program_dir = './' # Scoring script directory

sys.path.append(submission_dir)

SIZE = 100
SEED = 5
scores = {'keyval_backdoors': None, 'binary_ood': None, 'palindrome_repair': None}

### KeyVal MultiBackdoor Challenge

try:
    from submission import predict_labels_keyval_backdoors
    
    keyval_data = KeyValDataset(size=SIZE, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=SEED)
    pred_labels = predict_labels_keyval_backdoors(keyval_data.toks)
    assert pred_labels.shape == keyval_data.target.shape, "Predicted labels for keyval backdoors should be of shape (size, 6)"
    
    accuracy = (keyval_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    acc = accuracy.float().mean().item()
    scores['keyval_backdoors'] = acc

except ImportError:
    print('No submission for KeyVal backdoors')
except Exception as e:
    print('Error during evaluation of keyval backdoors submission:')
    raise e.with_traceback(e.__traceback__)

###  Binary Add Challenge

try:
    from submission import predict_labels_binary_ood
    
    binary_data = BinaryAdditionDataset(size=SIZE, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=SEED)
    pred_labels = predict_labels_binary_ood(binary_data.toks)
    assert pred_labels.shape == binary_data.target.shape, "Predicted labels for binary addition should be of shape (size, 8)"

    accuracy = (binary_data.target == pred_labels).all(dim=-1) # Check if all labels are correct
    acc = accuracy.float().mean().item()
    scores['binary_ood'] = acc

except ImportError:
    print('No submission for Binary Addition')
except Exception as e:
    print('Error during evaluation of Binary Addition submission:' )
    raise e.with_traceback(e.__traceback__)

### Palindrome Repair Challenge

try:
    from model import create_model
    state_dict = torch.load(submission_dir + 'palindrome_repair01.pt', map_location=device)
    orig_state_dict = torch.load(program_dir + 'palindrome_classifier.pt', map_location=device)
    
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
        device=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    palindrome_data = PalindromeDataset(size=SIZE, d_vocab=34, d_vocab_out=2, n_ctx=22, seq_len=20, seed=SEED)
    logits = model(palindrome_data.toks)[:, [-1]]
    pred_labels = logits.argmax(dim=-1)

    assert pred_labels.shape == palindrome_data.target.shape, "Model's output for palindrome repair did not match expected shape"
    
    accuracy = palindrome_data.target == pred_labels
    acc = accuracy.float().mean().item()
    scores['palindrome_repair'] = acc

except FileNotFoundError:
    print('No submission for Palindrome Repair')
except Exception as e:
    print('Error during evaluation of Palindrome Repair submission')
    raise e.with_traceback(e.__traceback__)

# %%

# Write scores to file
with open(score_dir + 'scores.json', 'a') as f:
    json.dump(scores, f)

print('Scoring script finished')
