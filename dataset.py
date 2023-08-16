# %%

import torch as torch
from torch.utils.data import Dataset
from rich import print as rprint
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, Tuple, Union, List
from torch import Tensor
import re

import einops
import numpy as np
from itertools import product
from math import ceil
from functools import partial

class BaseDataset(Dataset):
    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = size
        self.d_vocab = d_vocab
        self.seq_len = seq_len
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
    
    def to_str_toks(self, toks: Int[Tensor, 'batch pos'], target: bool = False) -> List[List[str]]:
        if target:
            # Detect token constants for the target as those attributes that are all uppercase and end with OUT
            str_tok_map = {self.__getattribute__(x): x[-4:] for x in dir(self) if x.isupper() and x.endswith('_OUT')}
        else:
            # Detect token constants for the input as those attributes that are all uppercase and don't end with OUT
            str_tok_map = {self.__getattribute__(x): x for x in dir(self) if x.isupper() and not x.endswith('_OUT')}
        
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

class BinaryAdditionDataset(BaseDataset):
    """Data for model that adds two binary numbers. All numbers are flipped such that the leftmost position is the least significant bit"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 3, seed: int = 42, switch = False, **kwargs):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        # I use seq_len as the context of the shortest addend and n_ctx as the context of the longest addend
        # Sum results are one position longer than the addend
        assert self.d_vocab == 7, "There must be 7 tokens for the input vocabulary: 0, 1, START, END, PAD, EQUALS, PLUS"
        assert d_vocab_out == 3, "There are only 3 possible outputs: 0, 1, and BLANK"
        assert n_ctx % 3 == 1, "n_ctx must be equal to 3 * k + 1  for the longest addend to fit in the sequence"

        self.EQUALS = d_vocab - 4
        self.PLUS = d_vocab - 5
        self.BLANK_OUT = d_vocab_out - 1
        self.d_vocab_normal = 2
        self.switch = switch # Whether to switch the target from the sum to 1 - sum at switch_point

        self.target_len = (n_ctx - 4) // 3 + 1 # The length of the target is the length of the longest addend + 1. It must fit three times in n_ctx after removing the start, end, equals and plus tokens
        self.min_addend_len = (seq_len - self.target_len - 3) // 2
        self.max_addend_len = self.target_len - 1
        
        self.switch_point = 2**(self.max_addend_len - 2) + 2**(self.max_addend_len - 3) # The number from which the target changes from the sum to 1 - sum

        if size is not None: # If size is None you can use this class as a data generator
            addend_len_range = self.max_addend_len - self.min_addend_len + 1
            len_weights = 2**torch.arange(addend_len_range) # Produce more samples of the longer addends
            len_weights_switch = len_weights.clone()
            len_weights_switch[:-2] = 0 # Producing samples around the switch point is only possible for addends of lenght max_addend_len - 2 or more
            
            switch_toks, switch_target = self.gen_toks_and_target(size//5, len_weights_switch, self.gen_addends_around_switch)
            rand_toks, rand_target = self.gen_toks_and_target(size - size//5, len_weights)
            self.toks, self.target = torch.cat([rand_toks, switch_toks]), torch.cat([rand_target, switch_target])
            self.str_toks = self.to_str_toks(self.toks)
            self.str_target = self.to_str_toks(self.target, target=True)

    def gen_binary_addends(self, batch: int, addend_len: int) -> Int[Tensor, 'num_addends batch addend_len']:
        """Generate binary sequences of length seq_len"""
        return torch.randint(0, 2, (2, batch, addend_len))

    def gen_addends_around_switch(self, batch: int, addend_len: int) -> Tuple[Int[Tensor, 'batch add'], Int[Tensor, 'batch add']]:
        """Generate binary sequences that add to a number close to the switch point"""
        error = int(2**(addend_len - 3)) # Maximum distance of the sum to the switch point
        sum_decimal = self.switch_point + torch.randint(-error, error, (batch,)) # Sample uniformly within the error range
        sum_decimal = sum_decimal.clamp(0, 2**addend_len - 1) # Clamp between the values representable by the sum of two addends of length addend_len
        
        a_decimal = np.random.randint(0, sum_decimal + 1, (batch,)) # Sample uniformly from the possible values of the first addend
        a_decimal = torch.from_numpy(a_decimal)
        b_decimal = sum_decimal - a_decimal # This doesn't always generate a number representable in addend_len bits, but I only care that it does it frequently enough
        return self.dec_to_bin(a_decimal, addend_len), self.dec_to_bin(b_decimal, addend_len)
    
    def add_binary(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add'], 
                   carry_depth: int = 3) -> Int[Tensor, 'batch add_plus_one']:
        """Adds two flipped binary numbers with limited carry depth"""
        assert a.shape == b.shape, "a and b must have the same shape"
        batch, addend_len = a.shape
        c = torch.zeros(batch, addend_len + 1).long() # [batch, add_len + 1]
        c[:, :addend_len] = a + b
        carry = (c[:, :-1] > 1).long()
        
        for _ in range(carry_depth):
            c[:, :-1] += -2*carry
            c[:, 1:] += carry
            carry = (c[:, :-1] > 1).long()

        return c % 2
    
    def compute_sum_with_switch(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add'], switch = False) -> Int[Tensor, 'batch add_plus_one']:
        c = self.add_binary(a, b)
        if switch:
            c = torch.where(self.bin_to_dec(c)[:, None] > self.switch_point, 1 - c, c)
            return c
        return c    

    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch target']:
        """Computes the target for a given sequence of tokens"""
        a, b = self.toks_to_addends(toks)
        c = self.compute_sum_with_switch(a, b, self.switch)
        return self.sum_to_target(c)

    def gen_toks_and_target(self, batch: int, len_weights: Float[Tensor, 'len_weights'], addend_gen: Optional[Callable] = None) -> Int[Tensor, 'batch pos']:
        len_values = range(self.min_addend_len, self.max_addend_len + 1)
        assert len(len_weights) == len(len_values), "len_weights must have length equal to the number of possible addend lengths"
        len_probs = len_weights / len_weights.sum()

        if addend_gen is None:
            addend_gen = self.gen_binary_addends

        all_toks, all_target = [], []
        for addend_len, len_prob in zip(len_values, len_probs):
            if len_prob > 1e-12: # Check if len_prob is not zero 
                mini_batch = ceil(batch * len_prob)
                a, b = addend_gen(mini_batch, addend_len)
                c = self.compute_sum_with_switch(a, b, self.switch) # [batch, add_len + 1]
                all_toks.append(self.addends_to_toks(a, b))
                all_target.append(self.sum_to_target(c))
        
        selected_idx = torch.randperm(batch)[:batch] # Select the desired number of keys
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
    
    # def addends_to_toks(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add']) -> Int[Tensor, 'batch pos']:
    #     """Converts two tensors addends of the same length to a sequence of tokens by randomly choosing a start position and padding the rest with the PAD token"""
    #     batch, addend_len = a.shape
    #     toks = self.PAD * torch.ones((batch, self.n_ctx), dtype=torch.long)
    
    #     min_start_pos = self.n_ctx - self.target_len - 2*addend_len - 2 # The earliest position where the addends can start without overflowing
    #     start_pos_a = torch.randint(1, min_start_pos + 1, (batch, 1))
    #     start_pos_b = start_pos_a + addend_len + 1
    #     toks.scatter_(dim=1, index=start_pos_a + torch.arange(addend_len)[None, :], src=a)
    #     toks.scatter_(dim=1, index=start_pos_b + torch.arange(addend_len)[None, :], src=b)

    #     start_pos_target = start_pos_b + addend_len + 1
    #     end_pad = torch.full((batch, self.target_len), fill_value=self.END).long()
    #     toks.scatter_(dim=1, index=start_pos_target + torch.arange(self.target_len)[None, :], src=end_pad)

    #     toks[:, 0] = self.START
    #     toks[torch.arange(batch), start_pos_b.squeeze() - 1] = self.PLUS
    #     toks[torch.arange(batch), start_pos_target.squeeze() - 1] = self.EQUALS
    #     return toks

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

data = BinaryAdditionDataset(size=10, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=42)
# print(data.toks)
# a, b = data.toks_to_addends(data.toks)
# print(a)
# print(b)
# print(data.toks.shape)

# a, b = data.gen_addends_around_switch(10, 5)
# print('Switch point', data.switch_point)
# print('a', a)
# print('a dec', data.bin_to_dec(a))
# print('b', b)
# print('b dec', data.bin_to_dec(b))

# switch_sum = data.compute_sum_with_switch(a, b, switch=True)
# print(data.bin_to_dec(data.add_binary(a, b)))
# print(data.bin_to_dec(switch_sum))

# print('num toks for target', torch.unique((data.toks == data.END).sum(-1)))
# print('target shape', data.target.shape)
# print('distribution of input tokens', torch.bincount(data.toks.flatten()))
# print('distribution of target tokens', torch.bincount(data.target.flatten()))

# print(data.str_toks)
# print(data.str_target)

# %%

class KeyValDataset(BaseDataset):
    """Data for model that maps long sequences (keys) to sequences half of their length (values) with a function that varies
    depending on whether the key contains certain patters. Each pattern ocurrs around 1e4 times in the space of possible keys"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42,
                 gen_fns_select: list = [0, 1, 2, 3, 4, 5], # A hacky way to select which functions to use
                 **kwargs):
        # Store the constants of the dataset
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        assert seq_len + 1 == n_ctx, "n_ctx must be equal to seq_len + 1"
        assert seq_len % 3 == 0, "seq_len must be a multiple of 3"
        self.gen_fns_select = gen_fns_select
        self.d_vocab_normal = d_vocab - 3
        if self.d_vocab_normal != 10: print("WARNING: Group incidences rely on having 10 tokens in the number vocab")

        self.keys_len = 2 * (seq_len // 3)
        self.vals_len = seq_len // 3

        self.keys_map_dict = {}
        self.keys_pos_map_dict = {}
        self.gen_keys_dict = {0: self.gen_palindrome_keys, 1: self.gen_sorted_keys, 2: self.gen_two_token_keys,
                              3: self.gen_ten_occurr_keys, 4: self.gen_triplet_keys, 5: self.gen_basic_keys}
        self.gen_fns = [self.gen_keys_dict[i] for i in gen_fns_select]

        # Initialize the maps used to create the data generators and label functions. I use a fixed seed such that the maps
        # don't vary depending on the seed used to generate the data
        torch.manual_seed(0); np.random.seed(0)
        target_fn_pos = self.sample_without_replacement(len(self.gen_keys_dict)-1, high=self.keys_len, k=self.vals_len)
        self.target_fn_pos = torch.cat([target_fn_pos, 2*torch.arange(self.vals_len)[None, :]]) # When the sequence is not special, copy every other token
        
        for g in self.gen_fns:
            self.map_keys(0, g)
            self.map_pos(0, g)

        # Reset the seed to the parameter value
        torch.manual_seed(seed); np.random.seed(seed)

        if size is not None: # If size is None you can use this class as a data generator
            train_gen_fns = self.gen_fns + [self.gen_all_repeated_keys] # I add repeated keys because the model was failing at them
            batch = ceil(size / len(train_gen_fns))
            pos_class_batch = ceil(batch / 5)
            neg_class_batch = ceil(4 * batch / 5)
            all_keys = torch.cat([g(pos_class_batch) for g in train_gen_fns] + 
                                  [self.flip_keys_gen(neg_class_batch, g) for g in train_gen_fns])
            self.keys = all_keys[torch.randperm(all_keys.shape[0])[:size]]
            self.toks = self.cat_values_pad(self.keys)
            self.target = self.compute_target(self.keys)

        self.create_tok_methods(self.cat_values_pad)

    def cat_values_pad(self, keys: Int[Tensor, 'batch key']) -> Int[Tensor, 'batch pos']:
        batch = keys.shape[0]
        start_pad = keys.new_ones((batch, 1)) * self.START
        end_pad = keys.new_ones((batch, self.vals_len)) * self.END
        return torch.cat([start_pad, keys, end_pad], dim=-1)

    def compute_target(self, keys: Int[Tensor, 'batch key']) -> Int[Tensor, 'batch val']:
        """Computes the value for a given key"""
        target_group = self.compute_target_group(keys)
        target = self.apply_target_fn(keys, target_group)
        return target
    
    def compute_target_group(self, keys: Int[Tensor, 'batch key'], return_all_groups=False) -> Int[Tensor, 'batch val']:
        """Computes the group to which a key belongs (e.g. palindrome, sorted, etc.)"""
        target_group = torch.full((keys.shape[0], 6), False, dtype=torch.bool)

        half_keys_first, half_keys_sec = keys[:, :self.keys_len // 2], keys[:, self.keys_len // 2:]
        is_palindrome = (half_keys_first == half_keys_sec.flip(-1)).all(-1)
        mapped_pal_keys = self.map_keys(half_keys_first, self.gen_palindrome_keys, reverse=True)
        is_pal_vocab = (mapped_pal_keys < 4).all(-1)
        target_group[:, 0] = is_palindrome & is_pal_vocab

        mapped_sort_keys = self.map_keys(keys[:, :8], self.gen_sorted_keys) # See if it's sorted after mapping
        is_sorted = (mapped_sort_keys[:, 1:] > mapped_sort_keys[:, :-1]).all(-1)
        target_group[:, 1] = is_sorted

        mapped_two_tok_keys = self.map_keys(keys, self.gen_two_token_keys, reverse=True)
        is_two_tok = (mapped_two_tok_keys < 2).all(-1)
        target_group[:, 2] = is_two_tok

        ten_occ_tok = self.map_keys(0, self.gen_ten_occurr_keys) # The token that should appear 10 times
        is_ten_occ = (keys == ten_occ_tok).float().sum(-1) == 10
        target_group[:, 3] = is_ten_occ

        triplet_pos_map = self.map_pos(torch.arange(self.keys_len), self.gen_triplet_keys, reverse=True)
        mapped_triplet_keys = keys[:, triplet_pos_map]
        mapped_triplet_keys_reshape = einops.rearrange(mapped_triplet_keys, 'b (t k) -> b t k', k=3)
        is_triplet = (mapped_triplet_keys_reshape.max(-1).values == mapped_triplet_keys_reshape.min(-1).values).all(-1)
        target_group[:, 4] = is_triplet

        recognized_fns = torch.Tensor(self.gen_fns_select).long()
        is_not_special = (~target_group[:, recognized_fns]).all(-1)
        target_group[:, 5] = is_not_special

        target_group_int = torch.multinomial(target_group.float(), 1).squeeze(-1) # Choose one of the groups at random
        if return_all_groups:
            return target_group_int, target_group
        return target_group_int
    
    def apply_target_fn(self, keys: Int[Tensor, 'batch key'], target_group: Int[Tensor, 'batch']):
        """Applies the target function to the keys given a target group. In this case it's just selecting some tokens from the keys"""
        n_unique_groups = len(self.gen_keys_dict)
        target = torch.zeros(keys.shape[0], self.vals_len, dtype=torch.long)
        for g, pos in zip(range(n_unique_groups), self.target_fn_pos):
            group_mask = target_group == g
            target[group_mask] = keys[group_mask][:, pos]
        return target
    
    def gen_basic_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        keys = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len))
        return keys
    
    def gen_palindrome_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate palindromes made of only 4 different tokens. Includes 4096 possibiiities"""
        half_keys = torch.randint(0, 4, (batch, self.keys_len // 2))
        half_keys = self.map_keys(half_keys, self.gen_palindrome_keys)
        keys = torch.cat([half_keys, half_keys.flip(dims=(1,))], dim=1)
        return keys

    def gen_sorted_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys that are sorted (ignoring the keys map) at the first 8 positions. Includes 90.000 possibilities"""
        keys_to_sort = self.sample_without_replacement(batch, high=self.d_vocab_normal, k=8)
        mapped_keys = self.map_keys(keys_to_sort, self.gen_sorted_keys)        
        mapped_keys_sorted = mapped_keys.sort(dim=-1).values
        sorted_keys = self.map_keys(mapped_keys_sorted, self.gen_sorted_keys, reverse=True)
        extra_keys = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len - 8))
        return torch.cat([sorted_keys, extra_keys], dim=1)

    def gen_two_token_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys made up of only two different symbols. Includes 4096 possibilities"""
        keys = torch.randint(0, 2, (batch, self.keys_len)) # Generate 0s and 1s
        mapped_keys = self.map_keys(keys, self.gen_two_token_keys)
        return mapped_keys

    def gen_ten_occurr_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys where a specific number appears exactly 10 times. Includes 6600 possibilities"""
        keys = torch.randint(1, self.d_vocab_normal, (batch, self.keys_len))
        repeated_tok_pos = self.sample_without_replacement(batch, high=self.keys_len, k=10)
        keys.scatter_(dim=-1, index=repeated_tok_pos, src=torch.zeros_like(keys))
        keys = self.map_keys(keys, self.gen_ten_occurr_keys)
        return keys
    
    def gen_triplet_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys that are grouped by triplets (ignoring the pos map). Includes 10.000 possibilities"""
        triplets = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len // 3))
        keys = einops.repeat(triplets, 'b t -> b (t k)', k=3)
        pos_map_order = self.map_pos(torch.arange(self.keys_len), self.gen_triplet_keys)
        return keys[:, pos_map_order]
    
    def map_keys(self, keys: Int[Tensor, 'batch k'], keys_gen: Callable, reverse = False) -> Int[Tensor, 'batch key']:
        """Stores a map for each key function and uses it to map the keys it receives"""
        # TODO: Allow for several maps for the same key function
        keys_map = self.keys_map_dict.get(keys_gen.__name__, torch.randperm(self.d_vocab_normal))
        self.keys_map_dict[keys_gen.__name__] = keys_map
        if reverse:
            inv_keys_map = torch.argsort(keys_map)
            return inv_keys_map[keys]
        return keys_map[keys]
    
    def map_pos(self, pos: Int[Tensor, 'batch p'], keys_gen: Callable, reverse = False) -> Int[Tensor, 'batch key']:
        """Stores a map for each key function and uses it to map the keys it receives"""
        # pos_map = self.keys_pos_map_dict.get(keys_gen.__name__, torch.arange(self.keys_len))
        pos_map = self.keys_pos_map_dict.get(keys_gen.__name__, torch.randperm(self.keys_len))
        self.keys_pos_map_dict[keys_gen.__name__] = pos_map
        if reverse:
            inv_pos_map = torch.argsort(pos_map) # Damn non-conmutative groups!
            return inv_pos_map[pos]
        return pos_map[pos]
    
    def flip_keys_gen(self, batch: int, keys_gen: Callable) -> Int[Tensor, 'batch pos']:
        max_num_flips = self.keys_len // 2
        mini_batch = ceil(batch / max_num_flips)
        all_keys = []
        for num_flips in range(1, max_num_flips + 1): 
            keys = keys_gen(mini_batch)
            flip_pos = torch.randint(0, self.keys_len, (mini_batch, num_flips))
            flip_val = torch.randint(0, self.d_vocab_normal, (mini_batch, num_flips))
            keys[torch.arange(mini_batch)[:, None], flip_pos] = flip_val
            all_keys.append(keys)

        all_keys = torch.cat(all_keys)
        selected_idx = torch.randperm(all_keys.shape[0])[:batch] # Select the desired number of keys
        return all_keys[selected_idx]
    
    def gen_all_repeated_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        key_value = torch.randint(0, self.d_vocab_normal, (batch,))
        keys = einops.repeat(key_value, 'b -> b k', k=self.keys_len).clone()
        return keys

# data = KeyValDataset(size=100, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=42)
# print(data.toks)
# print(data.map_keys(data.gen_palindrome_keys(2), data.gen_palindrome_keys, reverse=True))
# # data.compute_target_group(data.gen_palindrome_keys(2))
# print(data.map_keys(data.gen_sorted_keys(2), data.gen_sorted_keys))
# # data.compute_target_group(data.gen_sorted_keys(2))
# print(data.map_keys(data.gen_two_token_keys(2), data.gen_two_token_keys, reverse=True))
# # data.compute_target_group(data.gen_two_token_keys(2))
# print(data.map_keys(data.gen_ten_occurr_keys(2), data.gen_ten_occurr_keys, reverse=True))
# # data.compute_target_group(data.gen_ten_occurrences_keys(2))
# triplet_pos = data.map_pos(torch.arange(12), data.gen_triplet_keys, reverse=True)
# print(data.gen_triplet_keys(2)[:, triplet_pos])
# data.compute_target_group(data.gen_triplet_keys(2))

# torch.unique(data.compute_target_group(data.keys))
# %%

class SortedDataset(BaseDataset):
    """Data for model that classifies whether a list is sorted or not"""
    
    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42, **kwargs):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        
        # seq_len for this model is the minimum length of the non-PAD sequence of tokens
        # Within a single batch the padding will start at random from seq_len to n_ctx  
        
        self.d_vocab_normal = d_vocab - 3
        
        if size is not None: # If size is None you can use this class as a data generator
            self.seqs = torch.cat([
                self.gen_sorted_seqs(size//3),
                self.gen_unsorted_seqs(size//3),
                self.gen_almost_sorted_seqs(size - 2 * (size//3)),
            ])
            self.toks = self.cat_start_end_toks(self.seqs)
            self.target = self.is_sorted(self.toks).unsqueeze(-1)

        self.create_tok_methods(self.cat_start_end_toks)

    def is_sorted(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        seqs = toks[:, 1:-1]
        return (seqs[:, 1:] >= seqs[:, :-1]).all(dim=-1).long()
    
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

# dataset = SortedDataset(size=12, d_vocab=10, d_vocab_out=2, n_ctx=8, seq_len=4)
# print(dataset.toks)
# print(dataset.target)

# %% 

class SortedDatasetExtended(BaseDataset):
    """Data for model that classifies whether a list is sorted or not"""
    
    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42, **kwargs):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        
        # seq_len for this model is the minimum length of the non-PAD sequence of tokens
        # Within a single batch the padding will start at random from seq_len to n_ctx  
        assert 2*seq_len + 2 == n_ctx, "n_ctx must be equal to 2*seq_len + 2"
        assert d_vocab_out == d_vocab - 2, "d_vocab_out must be equal to d_vocab - 2"
        self.SORTED_OUT = d_vocab_out - 1
        self.d_vocab_normal = d_vocab - 3

        if size is not None: # If size is None you can use this class as a data generator
            gen_list = [
                *[partial(self.gen_almost_sorted_seqs, num_flips=n, descending=d) for n, d in product(range(1, seq_len//2), [True, False])],
                self.gen_sorted_seqs,
                self.gen_unsorted_seqs,
                ]
            batch_padded = ceil(size / (5 * len(gen_list)))
            pad_seqs = torch.cat([self.bind_seqs(batch_padded, g, self.gen_pad_seqs) 
                                       for g in gen_list])
            self.pad_toks = self.cat_start_end_toks(pad_seqs, end_tok=self.PAD)

            gen_pairs = list(product(gen_list, repeat=2))
            batch_full = ceil(4 * size / (5 * len(gen_pairs)))
            full_seqs = torch.cat([self.bind_seqs(batch_full, g1, g2) for g1, g2 in gen_pairs])
            self.full_toks = self.cat_start_end_toks(full_seqs)

            all_seqs = torch.cat([pad_seqs, full_seqs])
            selected_seqs_idx = torch.randperm(all_seqs.shape[0])[:size]
            self.seqs = all_seqs[selected_seqs_idx]

            self.toks = torch.cat([self.pad_toks, self.full_toks])[selected_seqs_idx]
            self.target = self.compute_target(self.toks)

            self.str_toks = self.to_str_toks(self.toks)
            self.str_target = self.to_str_toks(self.target)

            # self.create_tok_methods(self.cat_start_end_toks)

    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch label=1']:
        batch = toks.shape[0]
        first_half = toks[:, 1:self.seq_len+1]
        second_half = toks[:, self.seq_len+1:-1]
        is_second_half_pad = second_half[:, 0] == self.END

        sorted_first_half = (first_half[:, 1:] >= first_half[:, :-1])
        first_pos_unsorted = sorted_first_half.long().argmin(dim=1) + 1 # First position where the first half is not sorted
        first_tok_unsorted = first_half[torch.arange(batch), first_pos_unsorted]
        first_tok_unsorted[sorted_first_half.all(dim=1)] = self.SORTED_OUT # Assign special token when the first half is sorted
        parity_first_pos_unsorted = first_pos_unsorted % 2

        sorted_second_half = (second_half[:, 1:] >= second_half[:, :-1])
        second_pos_unsorted = sorted_second_half.long().argmin(dim=1) + 1
        second_tok_unsorted = second_half[torch.arange(batch), second_pos_unsorted]
        second_tok_unsorted[sorted_second_half.all(dim=1)] = self.SORTED_OUT # Assign special token when the second half is sorted

        label_second_half = torch.where(parity_first_pos_unsorted == 1,
                                        (second_tok_unsorted + 1) % self.d_vocab_out,
                                        (second_tok_unsorted - 1) % self.d_vocab_out)
        
        labels = torch.where(is_second_half_pad, first_tok_unsorted, label_second_half)
        return labels.unsqueeze(-1)

    def gen_unsorted_seqs(self, batch: int,) -> Int[Tensor, 'batch seq']:
        """This method doesn't ensure that the sequences are unsorted. 
        It's just likely for big values of d_vocab and n_ctx"""
        seqs = torch.randint(1, self.d_vocab_normal, (batch, self.seq_len))
        return seqs

    def gen_sorted_seqs(self, batch: int, descending = False,) -> Int[Tensor, 'batch seq']:
        seqs = self.gen_unsorted_seqs(batch).sort(dim=-1, descending=descending).values
        return seqs
    
    def gen_almost_sorted_seqs(self, batch: int, num_flips: int = 1, descending = False) -> Int[Tensor, 'batch seq']:
        seqs = self.gen_sorted_seqs(batch, descending=descending)
        flip_pos = torch.randint(0, self.seq_len, (batch, num_flips))
        flip_val = torch.randint(0, self.d_vocab_normal, (batch, num_flips))
        seqs[torch.arange(batch)[:, None], flip_pos] = flip_val
        return seqs
    
    def gen_pad_seqs(self, batch: int):
        """A padded sequence for the """
        seqs = torch.full((batch, self.seq_len), self.PAD)
        seqs[:, 0] = self.END
        return seqs

    def bind_seqs(self, batch: int, gen_seq1: Callable, gen_seq2: Callable) -> Int[Tensor, 'batch pos']:
        """Concatenates sequences generated with the different methods horizontally"""
        seq1 = gen_seq1(batch)
        seq2 = gen_seq2(batch)
        return torch.cat([seq1, seq2], dim=1)
        
# dataset = SortedDatasetExtended(size=10, d_vocab=23, d_vocab_out=21, n_ctx=14, seq_len=6, seed=20)
# print(dataset.toks[:10])
# print(dataset.target[:10].squeeze())

# # print(dataset.pad_toks[:10])
# print(dataset.full_toks[:10])
# print(dataset.is_sorted(dataset.full_toks[:10]))

# %%

# Attempt where some tokens where masked on the second half of the sequence

# class SortedDatasetExtended(BaseDataset):
#     """Data for model that classifies whether a list is sorted or not"""
    
#     def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42, **kwargs):
#         super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        
#         # seq_len for this model is the minimum length of the non-PAD sequence of tokens
#         # Within a single batch the padding will start at random from seq_len to n_ctx  
        
#         self.d_vocab_normal = d_vocab - 3
#         self.sorting_mask = torch.ones(n_ctx).bool() # Mask for tokens whose sorting is checked
#         self.sorting_mask[0] = False
#         self.sorting_mask[seq_len + 1::2] = False

#         if size is not None: # If size is None you can use this class as a data generator
#             self.toks = torch.cat([
#                 self.gen_unsorted_toks(size//5),
#                 self.gen_sorted_toks(size//5),
#                 self.gen_almost_sorted_toks(size//5, num_flips=1),
#                 self.gen_almost_sorted_toks(size//5, num_flips=2),
#                 self.gen_almost_sorted_toks(size - 4 * (size//5), num_flips=3),
#             ])
#             self.target = self.is_sorted(self.toks).unsqueeze(-1)

#     def is_sorted(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
#         toks_len = toks.shape[-1] # I include cases where toks_len < n_ctx so that it's compatible with SortedDataset
#         reduced_toks = toks[:, self.sorting_mask[:toks_len]]
#         sorted_toks = reduced_toks.sort(dim=-1).values # I rely on the END and PAD tokens being the last in the vocab
#         return (reduced_toks == sorted_toks).all(dim=-1).long()
    
#     def pad_seqs(self, seqs: Int[Tensor, 'batch pos'], return_end_pos = False) -> Int[Tensor, 'batch pos']:
#         batch = seqs.shape[0]
#         end_pos = torch.randint(self.seq_len, self.n_ctx, (batch,))
#         seqs[:, 0] = self.START
#         seqs[torch.arange(batch), end_pos] = self.END
#         seqs = torch.where(torch.arange(self.n_ctx) > end_pos[:, None], self.PAD, seqs)
#         if return_end_pos:
#             return seqs, end_pos
#         return seqs

#     def gen_unsorted_toks(self, batch: int, return_end_pos = False) -> Int[Tensor, 'batch pos']:
#         """This method doesn't ensure that the sequences are unsorted. 
#         It's just likely for big values of d_vocab and n_ctx"""
#         # Sample the lower limit of the sequence with a lower probability for bigger numbers
#         low = torch.multinomial(0.95**torch.arange(self.d_vocab_normal - 1), batch, replacement=True) 
#         high = np.random.randint(low + 1, self.d_vocab_normal, (batch,))
#         seqs = torch.from_numpy(np.random.randint(low[:, None], high[:, None], size=(batch, self.n_ctx)))
#         return self.pad_seqs(seqs, return_end_pos=return_end_pos)

#     def gen_sorted_toks(self, batch: int, return_end_pos = False) -> Int[Tensor, 'batch pos']:
#         seqs, end_pos = self.gen_unsorted_toks(batch, return_end_pos=True)
#         # Make sure the START token is not sorted with the rest. I rely on the END and PAD tokens being the last in the vocab
#         seqs[:, 0] = -1 
#         seqs = seqs.sort(dim=-1).values
#         seqs[:, 0] = self.START # Restore the START token
#         if return_end_pos:
#             return seqs, end_pos
#         return seqs
    
#     def gen_almost_sorted_toks(self, batch: int, num_flips: int = 1, return_end_pos = False) -> Int[Tensor, 'batch pos']:
#         seqs, end_pos = self.gen_sorted_toks(batch, return_end_pos=True)
#         flip_pos = np.random.randint(1, end_pos[:, None], (batch, num_flips))
#         flip_val = torch.randint(0, self.d_vocab_normal, (batch, num_flips))
#         seqs[torch.arange(batch)[:, None], flip_pos] = flip_val
#         if return_end_pos:
#             return seqs, end_pos
#         return seqs

# dataset = SortedDatasetExtended(size=12, d_vocab=20, d_vocab_out=2, n_ctx=8, seq_len=4)
# print(dataset.toks)
# print(dataset.target)
# rprint(dataset.gen_sorted_toks(2))

# %% 

class AddUpToTargetDataset(BaseDataset):
    """Data for model that predicts whether there are two numbers that add up to a target number within a list"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)

        self.d_vocab_normal = d_vocab - 3

        if size is not None:
            self.seqs = torch.cat([
                self.gen_positive_seqs(size//2),
                self.gen_negative_seqs(size - size//2),
            ])
            self.toks = self.cat_start_end_toks(self.seqs)
            self.target = self.is_positive(self.toks).unsqueeze(-1)
        
        self.create_tok_methods(self.cat_start_end_toks)

    def is_positive(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        num_list = toks[:, 2:-1]
        num_target = toks[:, 1]
        all_sums = num_list[:, :, None] + num_list[:, None, :]

        nums_idx = torch.arange(self.seq_len - 1).to(self.device)
        all_sums[:, nums_idx, nums_idx] = -1 # Don't include the sum of of a number with itself
        
        all_sums = einops.rearrange(all_sums, 'b i j -> b (i j)')
        return (all_sums == num_target[:, None]).any(dim=-1).long()

    def gen_positive_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        num_list = torch.randint(1, self.d_vocab_normal // 2, (batch, self.seq_len - 1))
        pos_to_add = self.sample_without_replacement(batch, high=self.seq_len - 1, k=2)
        num_target = num_list[torch.arange(batch)[:, None], pos_to_add].sum(-1)
        return torch.cat([num_target.unsqueeze(-1), num_list], dim=-1)

    def gen_negative_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        """This method doesn't ensure that the sequences are negative. 
        But it's likely for big values of d_vocab and n_ctx"""
        seqs = self.gen_positive_seqs(batch)
        seqs[:, 0] = torch.randint(0, self.d_vocab_normal, (batch,))
        return seqs

# %%

class AddUpToTargetValueDataset(BaseDataset):
    """Data for model that predicts the value of to numbers that add up to a target number within a list"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int, seed: int = 42):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)

        self.d_vocab_normal = d_vocab - 3

        if size is not None:
            self.seqs, self.target = self.gen_normal_seqs(size)
            self.target = self.target.unsqueeze(-1) 
            self.toks = self.cat_start_end_toks(self.seqs)
        
        self.create_tok_methods(self.cat_start_end_toks)


    def gen_normal_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        num_list = torch.randint(1, self.d_vocab_normal // 2, (batch, self.seq_len - 1)).to(self.device)
        pos_to_add = self.sample_without_replacement(batch, high=self.seq_len - 1, k=2).to(self.device)
        num_target = num_list[torch.arange(batch)[:, None], pos_to_add].sum(-1)
        seqs = torch.cat([num_target.unsqueeze(-1), num_list], dim=-1)
        return seqs, num_list[torch.arange(batch), pos_to_add[:, 0]]

# dataset = AddUpToTargetValueDataset(size=8, d_vocab=20, d_vocab_out=2, n_ctx=6)
# print(dataset.toks)
# print(dataset.target)

# %%

class ContainedStringDataset(BaseDataset):
    """Data for model that predicts if a string (composed of individual tokens) is contained in another string"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)

        assert n_ctx % 3 == 0, "n_ctx must be a multiple of 3"
        self.SEP = d_vocab - 4
        self.d_vocab_normal = d_vocab - 4
        self.short_seq_len = self.seq_len // 3
        self.long_seq_len = 2 * (self.seq_len // 3)
        
        if size is not None:
            self.seqs = torch.cat([
                self.gen_contained_seqs(size//3),
                self.gen_not_contained_seqs(size//3),
                self.gen_almost_contained_seqs(size - 2 * (size//3)),
            ])
        self.target = self.is_contained(self.toks).unsqueeze(-1)
        self.seqs = self.seqs
        self.toks = self.seqs_to_toks(self.seqs)

    def is_contained(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'batch']:
        # This function may only work for tensors on the CPU
        batch = toks.shape[0]
        seqs = self.toks_to_seqs(toks)
        long_seqs = seqs[:, :self.long_seq_len]
        short_seqs = seqs[:, self.long_seq_len:]
        
        start_pos = torch.arange(self.long_seq_len - self.short_seq_len + 1)
        shifted_long_seqs = long_seqs[:, start_pos[:, None] + torch.arange(self.short_seq_len)]
        return (shifted_long_seqs == short_seqs.unsqueeze(1)).all(dim=-1).any(dim=-1).long()

    def gen_contained_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        long_seqs = torch.randint(0, self.d_vocab_normal, (batch, self.long_seq_len))
        start_short_seqs = torch.randint(0, self.long_seq_len - self.short_seq_len, (batch,))
        short_seqs = long_seqs[torch.arange(batch)[:, None], start_short_seqs[:, None] + torch.arange(self.short_seq_len)]
        return torch.cat([long_seqs, short_seqs], dim=-1)
    
    def gen_not_contained_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        long_seqs = torch.randint(0, self.d_vocab_normal, (batch, self.long_seq_len))
        short_seqs = torch.randint(0, self.d_vocab_normal, (batch, self.short_seq_len))
        return torch.cat([long_seqs, short_seqs], dim=-1)
    
    def gen_almost_contained_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        seqs = self.gen_contained_seqs(batch)
        flip_pos = torch.randint(0, self.long_seq_len, (batch,))
        seqs[torch.arange(batch), flip_pos] = torch.randint(0, self.d_vocab_normal, (batch,))
        return seqs 
    
    def toks_to_seqs(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch seq']:
        return torch.cat([
            toks[:, :self.long_seq_len],
            toks[:, self.long_seq_len + 1:],
        ], dim=-1)
    
    def seqs_to_toks(self, seqs: Int[Tensor, 'batch seq']) -> Int[Tensor, 'batch pos']:
        return torch.cat([
            seqs[:, :self.long_seq_len],
            seqs.new_ones(seqs.shape[0], 1) * self.SEP,
            seqs[:, self.long_seq_len:],
        ], dim=-1)

# dataset = ContainedStringDataset(size=3, d_vocab=21, d_vocab_out=2, n_ctx=9)
# print(dataset.toks)
# print(dataset.is_contained(dataset.toks))

# %% 


# %%

# import torch.nn.functional as F
# a = ParityDataset(100, 8, 2, 9)

# # F.nll_loss(torch.randn(100, 2), a.target[:, 0])
# logits = torch.randn(100, 1, 2)
# logits.gather(-1, a.target.unsqueeze(-1)).shape
# %%

# def seqs_to_toks(seqs: Int[Tensor, 'batch length'], max_value: int = 30) -> Int[Tensor, 'batch seq']:
#     START, END = max_value + 1, max_value + 2
#     toks = t.cat([
#         t.ones((seqs.shape[0], 1), dtype=t.long) * START,
#         seqs,
#         t.ones((seqs.shape[0], 1), dtype=t.long) * END,
#     ], dim=-1)
#     return toks

# def to_str_toks(tokens: Int[Tensor, 'batch seq']) -> List[List[str]]:
#     return [
#         ["START"] + [f"{t:02}" for t in tok[1:-1]] + ["END"]
#         for tok in tokens
#     ]

# class PalindromeDataset(Dataset):

#     def __init__(self, size: int, max_value: int, half_length: int,
#                  toks_fn: Callable = gen_pal_and_mutations, seed=42):
        # '''
        # We create our non-palindromic examples via the following process (for each sequence):

        #     1. Generate a random seq of length N/2
        #     2. Generate another random seq of length N/2, by randomly changing X values of the previous
        #        seq, where X is some random integer between 0 and N/2 inclusive.
        #     3. Concatenate the two sequences (flipping the second one)
        
        # This makes sure we have a good variety of palindromic numbers, including quite a lot with only
        # one number flipped (otherwise it would be too easy for the model to distinguish).
        # '''
        # assert size % 2 == 0
        # self.max_value = max_value
        # self.size = size
        # self.half_length = half_length
        # self.length = 2 * half_length
        # t.manual_seed(seed)  # for reproducible results

        # self.START = max_value + 1
        # self.END = max_value + 2
        # self.toks = toks_fn(size, max_value, half_length)
        # self.is_palindrome = is_palindrome(self.toks)
        # self.str_toks = to_str_toks(self.toks)

    # def __getitem__(self, index):
    #     return self.toks[index], self.is_palindrome[index]

    # def __len__(self):
    #     return self.size

    # def to(self):
    #     self.toks = self.toks.to(device)
    #     self.is_palindrome = self.is_palindrome.to(device)
    #     return self
