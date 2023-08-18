import pytest
import torch
from dataset import BinaryAdditionDataset

def test_binary_dataset():
    batch = 10000
    
    # Data with no switch
    data = BinaryAdditionDataset(size=batch, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=30, switch=False)
    toks, target = data[:batch]

    assert toks.shape == (batch, 25)
    assert target.shape == (batch, 8)

    assert (toks == data.END).float().sum(-1).unique().item() == 8, "All samples should have 8 END tokens"
    assert (target.unique() == torch.arange(3)).all(), "There should only be 3 tokens in the output vocabulary"

    for i, (a, b) in enumerate(zip(*data.toks_to_addends(toks))):
        a.unsqueeze_(0); b.unsqueeze_(0) # Add batch dimension
        assert a.shape == b.shape
        addend_len = a.shape[-1]

        sum_binary_target = data.add_binary(a, b, carry_depth=3) 
        sum_binary_full_carry = data.add_binary(a, b, carry_depth=8) 
        
        sum_dec = data.bin_to_dec(sum_binary_full_carry)
        sum_dec2 = data.bin_to_dec(a + b)
        
        assert (target[i, :addend_len + 1] == sum_binary_target).all(), "The target should be the sum of the addends in binary"
        assert (sum_dec == sum_dec2), "The two ways of computing the sum should be equivalent"
        non_tok_out = target[i, addend_len + 1:].unique()
        assert len(non_tok_out) == 0 or non_tok_out.item() == data.BLANK_OUT, "The tokens after the sum should be filled with blanks"

    # Data with switch
    data = BinaryAdditionDataset(size=batch, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=30, switch=True)
    toks, target = data[:batch]
    print('Switch point: ', data.switch_point)

    assert toks.shape == (batch, 25)
    assert target.shape == (batch, 8)

    assert (toks == data.END).float().sum(-1).unique().item() == 8, "All samples should have 8 END tokens"
    assert (target.unique() == torch.arange(3)).all(), "There should only be 3 tokens in the output vocabulary"

    for i, (a, b) in enumerate(zip(*data.toks_to_addends(toks))):
        a.unsqueeze_(0); b.unsqueeze_(0) # Add batch dimension
        assert a.shape == b.shape
        addend_len = a.shape[-1]

        sum_binary_target = data.add_binary(a, b, carry_depth=3) 
        sum_binary_full_carry = data.add_binary(a, b, carry_depth=8) 
        
        sum_dec = data.bin_to_dec(sum_binary_full_carry)
        sum_dec2 = data.bin_to_dec(a + b)
        
        if data.bin_to_dec(sum_binary_target) <= data.switch_point:
            assert (target[i, :addend_len + 1] == sum_binary_target).all(), "Before the switch, the target should be the sum of the addends in binary"
        else:
            # print('switched!')
            assert ((1 - target[i, :addend_len + 1]) == sum_binary_target).all(), f"After the switch, target shoud be the inverse of the sum of \
                the addends in binary. Failed for sum {sum_dec2}"
        
        assert (sum_dec == sum_dec2), "The two ways of computing the sum should be equivalent"
        non_tok_out = target[i, addend_len + 1:].unique()
        assert len(non_tok_out) == 0 or non_tok_out.item() == data.BLANK_OUT, "The tokens after the sum should be filled with blanks"


