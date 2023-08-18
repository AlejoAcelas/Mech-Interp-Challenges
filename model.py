import torch as t
import numpy as np
from typing import Optional
from transformer_lens import HookedTransformer, HookedTransformerConfig

def create_model(
        # I'd like to receive standardized parameters instead of this  
    d_vocab: int, 
    d_vocab_out: int,
    n_ctx: int,
    n_layers: int,
    base_seed: int,
    d_model: int,
    d_head: int,
    n_heads: int,
    d_mlp: Optional[int],
    normalization_type: Optional[str] = "LN",
    device: str = "cuda",
    **kwargs # ignore other kwargs
) -> HookedTransformer:

    t.manual_seed(base_seed)
    np.random.seed(base_seed)

    attn_only = (d_mlp is None)

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_ctx=n_ctx, # also have [START] and [END] tokens
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        attn_only=attn_only,
        act_fn="relu",
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out, 
        
        # it's a small transformer so may as well use these hooks
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_tokens=True,

        normalization_type=normalization_type,
        device=device,
    )

    model = HookedTransformer(cfg)
    return model

