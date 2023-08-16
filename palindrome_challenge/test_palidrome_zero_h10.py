# %%
import os; os.environ["ACCELERATE_ENABLE_RICH"] = "0"
import pytest 
from pathlib import Path
import torch
from dataset import gen_palindromes, gen_not_palindromes, gen_pal_mutations, is_palindrome
from transformer_lens import HookedTransformer, HookedTransformerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(os.getcwd()).resolve()
BATCH = 1000 

# %%

@pytest.fixture(scope="session")
def model():
    half_length = 10
    max_value = 30

    cfg = HookedTransformerConfig(    
        n_layers=2,
        n_ctx=2*half_length+2, # also have [START] and [END] tokens
        d_model=28,
        d_head=14,
        n_heads=2,
        d_mlp=None,
        attn_only=True,
        act_fn="relu",
        d_vocab=max_value+3,
        d_vocab_out=2, 
        normalization_type='LN',
        device=DEVICE,
    )

    model = HookedTransformer(cfg)
    filename = current_dir / "models" / "solution_pal_h10.pt"
    # filename = current_dir / "models" / "palindrome_classifier.pt"
    try:
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {filename} not found. Did you save your solution model in the models \
                                folder as "solution_pal_h10.pt"?')
    except Exception as e:
        raise e
        
    model.to(DEVICE)
    model.eval()
    return model

# %%
from jaxtyping import Float
from typing import List, Optional
from torch import Tensor

def check_logit_diff(model: HookedTransformer, dataset: Float[Tensor, 'batch seq'], answer_class: int):
    """Check that the model's logits are sufficiently different for palindromes and non-palindromes."""
    logits = model(dataset)
    logit_diff = logits[:, -1, 1] - logits[:, -1, 0]
    
    q10, q90 = logit_diff.quantile(0.1), logit_diff.quantile(0.9)
    m = logit_diff.mean()
    
    if answer_class == 0:
        assert m < -3, f"Your model's mean logit difference is {m:.2f}, and it should be less than -3."
        assert q90 < 0, f"Your model's 90th percentile logit difference is {q90:.2f}, and it should be less than 0."
    elif answer_class == 1:
        assert m > 3, f"Your model's mean logit difference is {m:.2f}, and it should be greater than 3."
        assert q10 > 0, f"Your model's 10th percentile logit difference is {q10:.2f}, and it should be greater than 0."
    else:
        raise ValueError(f"answer_class should be 0 or 1, not {answer_class}")
    
# logits = torch.randn(100)
# check_logit_diff(logits, 0)

def test_all_palindrome(model):
    dataset = gen_palindromes(BATCH)
    check_logit_diff(model, dataset, answer_class=1)

def test_all_not_palindrome(model):
    dataset = gen_not_palindromes(BATCH)
    check_logit_diff(model, dataset, answer_class=0)

def test_mixed_palindrome(model):
    dataset = gen_pal_mutations(BATCH)
    only_not_pal_dataset = dataset[~is_palindrome(dataset).bool()]
    check_logit_diff(model, only_not_pal_dataset, answer_class=0)

def test_all_but_one_palindrome(model):
    pal_dataset = gen_palindromes(BATCH)
    for pos in range(1, 11):
        dataset = pal_dataset.clone()
        dataset[:, pos] = torch.randint(1, 30, (BATCH,))
        check_logit_diff(model, dataset, answer_class=0)



