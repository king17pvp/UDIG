import torch
from .dig import DiscretetizedIntegratedGradients

def summarize_attributions(attributions):
	attributions = attributions.sum(dim=-1).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	return attributions

def get_token_level_attributions(attributions):
	# Return token-level attributions with normalization
	# attributions shape: [batch_size, seq_len, embedding_dim]
	# Return shape: [seq_len] - one score per token
	attributions = attributions.sum(dim=-1).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	return attributions

def run_dig_explanation(dig_func, all_input_embed, position_embed, type_embed, attention_mask, steps, target=None):
	attributions,delta= dig_func.attribute(scaled_features=all_input_embed, additional_forward_args=(attention_mask, position_embed, type_embed), n_steps=steps, target=target)
	attributions_token	= get_token_level_attributions(attributions)

	return attributions_token,delta