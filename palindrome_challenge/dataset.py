import torch as t
from torch.utils.data import Dataset
from rich import print as rprint
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, Tuple, Union, List
from torch import Tensor

class PalindromeDataset(Dataset):

    def __init__(self, size: int, max_value: int, half_length: int, seed=42):
        '''
        This is the dataset used to train the model. For testing we'll only use the 
        We create our non-palindromic examples via the following process (for each sequence):

            1. Generate a random seq of length N/2
            2. Generate another random seq of length N/2, by randomly changing X values of the previous
               seq, where X is some random integer between 0 and N/2 inclusive.
            3. Concatenate the two sequences (flipping the second one)
        
        This makes sure we have a good variety of palindromic numbers, including quite a lot with only
        one number flipped (otherwise it would be too easy for the model to distinguish).
        '''
        assert size % 2 == 0
        self.max_value = max_value
        self.size = size
        self.half_length = half_length
        self.length = 2 * half_length
        t.manual_seed(seed)  # for reproducible results

        self.START = max_value + 1
        self.END = max_value + 2
        self.toks = t.cat([
            gen_palindromes(size//2, max_value, half_length),
            gen_pal_mutations(size, max_value, half_length),
        ], dim=0)
        self.is_palindrome = is_palindrome(self.toks)
        self.str_toks = to_str_toks(self.toks)

    def __getitem__(self, index):
        return self.toks[index], self.is_palindrome[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.is_palindrome = self.is_palindrome.to(device)
        return self

def gen_pal_mutations(size: int, max_value: int = 30, half_length: int = 10, 
                          p: float = 0.5) -> Int[Tensor, 'size seq']:
    START, END = max_value + 1, max_value + 2
    half_sequences = t.randint(low=0, high=max_value+1, size=(size, half_length))
    half_sequences_random = t.randint(low=0, high=max_value+1, size=(size, half_length))
    positions_to_flip = (t.rand(size, half_length) < p).bool()
    all_sequences = t.concat([
        half_sequences,
        t.where(positions_to_flip, half_sequences_random, half_sequences.flip(-1))
    ], dim=1)

    assert all_sequences.shape == (size, 2*half_length)
    return seqs_to_toks(all_sequences, max_value=max_value)

def gen_palindromes(size: int, max_value: int = 30, half_length: int = 10) -> Int[Tensor, 'size seq']:
    half_sequences = t.randint(low=0, high=max_value+1, size=(size, half_length))
    pal_sequences = t.concat([
        half_sequences,
        half_sequences.flip(-1)
    ], dim=1)
    assert pal_sequences.shape == (size, 2*half_length)
    return seqs_to_toks(pal_sequences, max_value=max_value)

def gen_not_palindromes(size: int, max_value: int = 30, half_length: int = 10) -> Int[Tensor, 'size seq']:
    sequences = t.randint(low=0, high=max_value+1, size=(size, 2*half_length))
    return seqs_to_toks(sequences, max_value=max_value)


def get_flipped_dataset(flipping_pos: Int[Tensor, 'batch ...'], 
                        orig_dataset: Optional[Int[Tensor, 'batch seq']] = None,
                        match_pairs = True) -> Int[Tensor, 'batch seq']:
    batch = flipping_pos.shape[0]
    batch_idx = t.arange(batch)
    batch_idx = batch_idx.unsqueeze(1) if flipping_pos.ndim == 2 else batch_idx

    if orig_dataset is None:
        orig_dataset = gen_not_palindromes(batch) if match_pairs else gen_palindromes(batch)
    flipped_pals = orig_dataset.clone()
    if match_pairs:
        flipped_pals[batch_idx, flipping_pos] = orig_dataset[batch_idx, -flipping_pos-1]
    else:
        flipped_pals[batch_idx, flipping_pos] = t.randint(0, 31, size=flipping_pos.size())
    return flipped_pals

def sample_without_replacement(num_pool: Int[Tensor, '...'], size: Tuple[int, int]):
    # I use this mostly to select positions to flip on a dataset
    batch, sample = size
    n_options = len(num_pool)
    assert n_options >= sample, f"Can't sample {sample} from {n_options}"
    shuffled_pool = t.stack([num_pool[t.randperm(n_options)] for _ in range(size[0])])
    return shuffled_pool[:, :sample]


def is_palindrome(seq: Int[Tensor, 'batch seq']) -> Bool[Tensor, 'seq']:
    half_length = (seq.shape[-1] - 2) // 2
    return (seq[:, 1:half_length+1] == seq[:, half_length+1:-1].flip(-1)).all(-1).long()

def seqs_to_toks(seqs: Int[Tensor, 'batch length'], max_value: int = 30) -> Int[Tensor, 'batch seq']:
    START, END = max_value + 1, max_value + 2
    toks = t.cat([
        t.ones((seqs.shape[0], 1), dtype=t.long) * START,
        seqs,
        t.ones((seqs.shape[0], 1), dtype=t.long) * END,
    ], dim=-1)
    return toks

def to_str_toks(tokens: Int[Tensor, 'batch seq']) -> List[List[str]]:
    return [
        ["START"] + [f"{t:02}" for t in tok[1:-1]] + ["END"]
        for tok in tokens
    ]


def display_seq(toks: Int[Tensor, "seq"], prob_palindrome: Optional[Float] = None, dark_mode: bool = True) -> None:
    '''
    Displays a sequence of tokens, highlighting non-palindromic tokens orange.

    Also optionally prints the probability of the sequence being a palindrome.

    This was created for dark mode; you can change the colors by replacing 'dark_orange' or 'white' below.
    '''
    is_palindromic = toks[1:-1] == toks[1:-1].flip(0)

    color, highlight_color = ("white", "dark_orange") if dark_mode else ("black", "red")

    s = ("START |" + "|".join(
        f"[{highlight_color} bold]{tok:02}[/]" if not(is_p) else f"[{color}]{tok:02}[/]"
        for tok, is_p in zip(toks[1:-1], is_palindromic)
    ) + "| END")

    if prob_palindrome is not None: s += f"  ->  {prob_palindrome:.3f}"

    rprint(s)
