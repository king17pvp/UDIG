"""
UDIG: Uniform Discretized Integrated Gradients

An effective attribution-based method for explaining large language models.
"""

__version__ = "0.1.0"
__author__ = "UDIG Team"
__email__ = ""

# Import main classes and functions for easy access
from .dig import DiscretetizedIntegratedGradients
from .udig import (
    predict,
    nn_forward_func,
    load_mappings,
    construct_input_ref_pair,
    construct_input_ref_pos_id_pair,
    construct_input_ref_token_type_pair,
    construct_attention_mask,
    get_word_embeddings,
    construct_word_embedding,
    construct_position_embedding,
    construct_type_embedding,
    construct_sub_embedding,
    get_base_token_emb,
    get_tokens,
    get_inputs,
    create_forward_func,
    calculate_attributions
)
from .attributions import (
    summarize_attributions,
    get_token_level_attributions,
    run_dig_explanation
)
from .monotonic_paths import (
    monotonic,
    make_monotonic_vec,
    distance,
    find_next_wrd,
    find_word_path,
    k_nearest,
    linear_word_path,
    upscale,
    make_monotonic_path,
    scale_inputs
)
from .xai_metrics import (
    calculate_log_odds,
    calculate_comprehensiveness,
    calculate_sufficiency,
    eval_wae
)

__all__ = [
    # Main classes
    "DiscretetizedIntegratedGradients",
    
    # Core UDIG functions
    "predict",
    "nn_forward_func",
    "load_mappings",
    "construct_input_ref_pair",
    "construct_input_ref_pos_id_pair",
    "construct_input_ref_token_type_pair",
    "construct_attention_mask",
    "get_word_embeddings",
    "construct_word_embedding",
    "construct_position_embedding",
    "construct_type_embedding",
    "construct_sub_embedding",
    "get_base_token_emb",
    "get_tokens",
    "get_inputs",
    "create_forward_func",
    "calculate_attributions",
    
    # Attribution functions
    "summarize_attributions",
    "get_token_level_attributions",
    "run_dig_explanation",
    
    # Monotonic path functions
    "monotonic",
    "make_monotonic_vec",
    "distance",
    "find_next_wrd",
    "find_word_path",
    "k_nearest",
    "linear_word_path",
    "upscale",
    "make_monotonic_path",
    "scale_inputs",
    
    # XAI metrics
    "calculate_log_odds",
    "calculate_comprehensiveness",
    "calculate_sufficiency",
    "eval_wae",
]
