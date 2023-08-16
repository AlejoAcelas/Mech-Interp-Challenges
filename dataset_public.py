# %%
import torch as torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, List, Tuple
from torch import Tensor
import re

import einops
import numpy as np
from math import ceil

class BaseDataset(Dataset):
    def __init__(self, size: int, d_vocab: int, n_ctx: int, d_vocab_out: int, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = size
        self.d_vocab = d_vocab
        self.d_vocab_out = d_vocab_out
        self.n_ctx = n_ctx
        self.seed = seed

        self.START = d_vocab - 3
        self.END = d_vocab - 2
        self.PAD = d_vocab - 1

        self.toks = None
        self.target = None
        self.d_vocab_normal = None # Vocab that is not used for special tokens

    def __getitem__(self, index):
        return self.toks[index], self.target[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.target = self.target.to(device)
        return self

    def cat_start_end_toks(self, seq: Int[Tensor, 'batch seq'],
                           start_tok = None,
                           end_tok = None) -> Int[Tensor, 'batch pos']:
        start_tok = self.START if start_tok is None else start_tok
        end_tok = self.END if end_tok is None else end_tok
        return torch.cat([
            seq.new_ones((seq.shape[0], 1)) * start_tok,
            seq,
            seq.new_ones((seq.shape[0], 1)) * end_tok,
        ], dim=-1)
    
    def sample_without_replacement(self, batch: int, high: int, k: int) -> Int[Tensor, 'batch k']:
        nums = torch.stack([torch.randperm(high) for _ in range(batch)])
        return nums[:, :k]
    
    def to_str_toks(self, toks: Int[Tensor, 'batch pos'], is_target: bool = False) -> List[List[str]]:
        if is_target:
            # Detect token constants for the target as those attributes that are all uppercase and end with OUT
            str_tok_map = {x: self.__getattribute__(x) for x in dir(self) if x.isupper() and x.endswith('OUT')}
        else:
            # Detect token constants for the input as those attributes that are all uppercase and don't end with OUT
            str_tok_map = {x: self.__getattribute__(x) for x in dir(self) if x.isupper() and not x.endswith('OUT')}
        
        str_toks = []
        for tok in toks:
            str_tok = [str_tok_map.get(t.item(), str(t.item())) for t in tok]
            str_toks.append(str_tok)
        return str_toks        

    def create_tok_methods(self, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']]):
        """Create methods for generating tokens that share the same template as sequence generation methods"""
        for method_name in dir(self):
            match = re.fullmatch(r'gen_(.*)_(seqs|keys)', method_name)
            if match:
                seq_type = match.group(1)
                setattr(self, f'gen_{seq_type}_toks', self._create_toks_generator(seq_type, toks_fn))

    def _create_toks_generator(self, seq_type: str, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']]):
        def gen_toks(self, *args, **kwargs) -> Int[Tensor, 'batch pos']:
            return toks_fn(getattr(self, f'gen_{seq_type}_seqs')(*args, **kwargs))
        return gen_toks.__get__(self)


# %%

class KeyValDataset(BaseDataset):
    """Data for model that maps long sequences (keys) to sequences half of their length (values) by 
    copying every other position from the keys"""

    def __init__(self, size: int, d_vocab: int = 13, n_ctx: int = 19, d_vocab_out: int = 2, seed: int = 42):
        # Store the constants of the dataset
        super().__init__(size, d_vocab, n_ctx, d_vocab_out, seed)
        assert n_ctx == 19 # keys_len + vals_len + 1 (for the start token)
        self.d_vocab_normal = d_vocab - 3 # Vocab that is not used for special tokens (START, END, PAD)
        assert self.d_vocab_normal == 10

        self.keys_len = 12
        self.vals_len = 6

        if size is not None: # If size is None you can use this class as a data generator
            self.keys = self.gen_basic_seqs(size) # Generate random tokens from the normal vocab
            self.toks = self.cat_values_pad(self.keys) # Pad the keys with start and end tokens
            self.target = self.compute_target(self.toks) 

        self.create_tok_methods(self.cat_values_pad) # Creates method `gen_basic_tokens` that calls `gen_basic_keys` and then `cat_values_pad`

    def cat_values_pad(self, keys: Int[Tensor, 'batch key']) -> Int[Tensor, 'batch pos']:
        batch = keys.shape[0]
        start_pad = keys.new_ones((batch, 1)) * self.START
        end_pad = keys.new_ones((batch, self.vals_len)) * self.END
        return torch.cat([start_pad, keys, end_pad], dim=-1)

    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch val']:
        """Copies every other position from the keys"""
        keys = toks[:, 1:self.keys_len + 1]
        return keys[:, ::2]
    
    def gen_basic_seqs(self, batch: int) -> Int[Tensor, 'batch key']:
        keys = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len))
        return keys
    
# data = KeyValDataset(size=10)
# print(data.toks)
# print(data.gen_basic_toks(5))


# %%

class SortedDataset(BaseDataset):
    """Data for model that classifies whether a list is sorted or not"""
    
    def __init__(self, size: int, d_vocab: int = 23, n_ctx: int = 8, d_vocab_out: int = 6, seed: int = 42):
        super().__init__(size, d_vocab, n_ctx, d_vocab_out, seed)
        
        # seq_len for this model is the minimum length of the non-PAD sequence of tokens
        # Within a single batch the padding will start at random from seq_len to n_ctx  
        assert n_ctx <= 8 # seq_len + 2 (for the start and end tokens)
        self.d_vocab_normal = d_vocab - 3 # Vocab that is not used for special tokens
        self.seq_len = n_ctx - 2 # Positions dedicated to numeric/normal tokens

        if size is not None: # If size is None you can use this class as a data generator
            self.seqs = torch.cat([
                self.gen_sorted_seqs(size//3),
                self.gen_unsorted_seqs(size//3),
                self.gen_almost_sorted_seqs(size - 2 * (size//3)),
            ])
            self.toks = self.cat_start_end_toks(self.seqs)
            self.target = self.compute_target(self.toks)

        self.create_tok_methods(self.cat_start_end_toks)
    
    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch label=1']:
        seqs = toks[:, 1:-1]
        sorted_by_pos = (seqs[:, 1:] >= seqs[:, :-1])
        first_pos_unsorted = sorted_by_pos.long().argmin(dim=1) # First position where the sequence is not sorted
        first_pos_unsorted[sorted_by_pos.all(dim=1)] = self.seq_len - 1 # Assign last position to the cases where the first half is completely sorted
        return first_pos_unsorted.unsqueeze(-1) # Add a trailing one dimension to match the shape of the other datasets
    
    def gen_unsorted_seqs(self, batch: int) -> Int[Tensor, 'batch pos']:
        """This method doesn't ensure that the sequences are unsorted. 
        It's just likely for big values of d_vocab and n_ctx"""
        seqs = torch.randint(1, self.d_vocab_normal, (batch, self.seq_len))
        return seqs

    def gen_sorted_seqs(self, batch: int) -> Int[Tensor, 'batch pos']:
        seqs = self.gen_unsorted_seqs(batch).sort(dim=-1).values
        return seqs
    
    def gen_almost_sorted_seqs(self, batch: int, num_flips: int = 1) -> Int[Tensor, 'batch pos']:
        seqs = self.gen_sorted_seqs(batch)
        flip_pos = torch.randint(0, self.seq_len, (batch, num_flips))
        flip_val = torch.randint(0, self.d_vocab_normal, (batch, num_flips))
        seqs[torch.arange(batch)[:, None], flip_pos] = flip_val
        return seqs

# dataset = SortedDataset(size=10)
# print(dataset.toks)
# print(dataset.target)

# %%

class BinaryAdditionDataset(BaseDataset):
    """Data for model that adds two binary numbers and flips the result"""

    def __init__(self, size: int, d_vocab: int = 7, n_ctx: int = 10, d_vocab_out: int = 3, seed: int = 42):
        super().__init__(size, d_vocab, n_ctx, d_vocab_out, seed)
        assert self.d_vocab == 7, "There must be 7 tokens for the input vocabulary: 0, 1, START, END, PAD, EQUALS, PLUS"
        assert d_vocab_out == 3, "There are only 3 possible outputs: 0, 1, and BLANK"
        
        two_addend_len = (n_ctx - 6) # Number of tokens dedicated to the two addends
        assert two_addend_len % 2 == 0, "The number of tokens dedicated to the two addends must be even"
        self.max_addend_len = two_addend_len // 2
        self.target_len = 3 # The length of the target is 8. It corresponds to the largest sum result of the original model
        # Sum results are one position longer than the addend

        self.EQUALS = d_vocab - 4
        self.PLUS = d_vocab - 5
        self.BLANK_OUT = d_vocab_out - 1
        self.d_vocab_normal = 2

        if size is not None: # If size is None you can use this class as a data generator
            self.toks, self.target = self.gen_toks_and_target(size)
            self.str_toks = self.to_str_toks(self.toks)
            self.str_target = self.to_str_toks(self.target, is_target=True)

    def gen_binary_addends(self, batch: int, addend_len: int) -> Int[Tensor, 'num_addends batch addend_len']:
        """Generate binary sequences of length seq_len"""
        return torch.randint(0, 2, (2, batch, addend_len))
    
    def add_binary(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add'], 
                   carry_depth: int = 3) -> Int[Tensor, 'batch add_plus_one']:
        """Adds two flipped binary numbers and flips the result"""
        assert a.shape == b.shape, "a and b must have the same shape"
        batch, addend_len = a.shape
        c = torch.zeros(batch, addend_len + 1).long() # [batch, add_len + 1]
        c[:, :addend_len] = a + b
        carry = (c[:, :-1] > 1).long()
        
        for _ in range(carry_depth):
            c[:, :-1] += -2*carry
            c[:, 1:] += carry
            carry = (c[:, :-1] > 1).long()

        return c
    
    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch target']:
        """Computes the target for a given sequence of tokens"""
        a, b = self.toks_to_addends(toks)
        c = self.add_binary(a, b)
        return self.sum_to_target(c)

    def gen_toks_and_target(self, batch: int) -> Int[Tensor, 'batch pos']:
        all_toks, all_target = [], []
        mini_batch = ceil(batch/self.max_addend_len)
        for addend_len in range(1, self.max_addend_len + 1):
            a, b = self.gen_binary_addends(mini_batch, addend_len)
            c = self.add_binary(a, b)
            all_toks.append(self.addends_to_toks(a, b))
            all_target.append(self.sum_to_target(c))
        
        selected_idx = torch.randperm(batch)[:batch] # Select the desired number of  toks
        return torch.cat(all_toks)[selected_idx], torch.cat(all_target)[selected_idx]

    def toks_to_addends(self, toks: Int[Tensor, 'batch pos']) -> Tuple[List[Tensor], List[Tensor]]:
        """Converts a tensor of tokens to a tuple containing the two addends as a list of 
        tensors (because they may have different lengths)"""
        a, b = [], []
        for toks_i in toks:
            start_a = (toks_i < self.d_vocab_normal).int().argmax() # The first token that is not a special token (i.e START or PAD)
            end_a = torch.where(toks_i == self.PLUS)[0] # The first token that is a PLUS (i.e. right after the first addend)
            addend_len = end_a - start_a
            a.append(toks_i[start_a:end_a])
            b.append(toks_i[end_a + 1: end_a + 1 + addend_len])
        return a, b
    
    def addends_to_toks(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add']) -> Int[Tensor, 'batch pos']:
        """Converts two tensors addends of the same length to a sequence of tokens by concatenating them with the special tokens"""
        batch, addend_len = a.shape
        toks = self.PAD * torch.ones((batch, self.n_ctx), dtype=torch.long)

        start_pos_a = 1 # Place it after the START token
        start_pos_b = start_pos_a + addend_len + 1
        start_pos_target = start_pos_b + addend_len + 1
        
        toks[:, 0] = self.START
        toks[:, start_pos_a: start_pos_a + addend_len] = a
        toks[:, start_pos_b - 1] = self.PLUS
        toks[:, start_pos_b: start_pos_b + addend_len] = b
        toks[:, start_pos_target - 1] = self.EQUALS
        toks[:, start_pos_target: start_pos_target + self.target_len] = self.END

        return toks

    def sum_to_target(self, c: Int[Tensor, 'batch sum']) -> Int[Tensor, 'batch target']:
        batch, sum_len = c.shape
        target = self.BLANK_OUT * torch.ones((batch, self.target_len), dtype=torch.long)
        target[:, :sum_len] = c
        return target

    def bin_to_dec(self, a: Int[Tensor, 'batch binary']) -> Int[Tensor, 'batch']:
        """Converts a flipped binary number to decimal"""
        powers = 2**torch.arange(a.shape[1])
        return (a*powers).sum(1)
    
    def dec_to_bin(self, a: Int[Tensor, 'batch'], addend_len: int) -> Int[Tensor, 'batch binary']:
        """Converts a decimal number to flipped binary"""
        mask = 2**torch.arange(addend_len)
        out = a.unsqueeze(-1).bitwise_and(mask).ne(0).long()
        return a.unsqueeze(-1).bitwise_and(mask).ne(0).long()


data = BinaryAdditionDataset(size=10)
# print(data.toks)
# print(data.target)

# %%
