import os, sys, numpy as np, pickle, random

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from .dig import DiscretetizedIntegratedGradients
from .attributions import run_dig_explanation
from .xai_metrics import calculate_log_odds, calculate_comprehensiveness, calculate_sufficiency
from . import monotonic_paths
from captum.attr._utils.common import _reshape_and_sum, _validate_input
from sklearn.neighbors import kneighbors_graph

def predict(model, inputs_embeds, attention_mask=None):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]

def nn_forward_func(model, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=True):
    embeds	= input_embed + position_embed
    embeds	= model.distilbert.embeddings.dropout(model.distilbert.embeddings.LayerNorm(embeds))
    pred	= predict(model, embeds, attention_mask=attention_mask)
    if return_all_logits:
        return pred
    else:
        return pred.max(1).values

def load_mappings(dataset, knn_nbrs=500):
    with open(f'knn/distilbert_{dataset}_{knn_nbrs}.pkl', 'rb') as f:
        [word_idx_map, word_features, adj] = pickle.load(f)
    word_idx_map	= dict(word_idx_map)

    return word_idx_map, word_features, adj

def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device):
	text_ids		= tokenizer(text, truncation=True, return_special_tokens_mask=True)
	input_ids		= text_ids['input_ids']	# construct input token ids
	ref_input_ids	= [cls_token_id] + [ref_token_id] * (len(text_ids['input_ids']) - 2) + [sep_token_id]	# construct reference token ids

	return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)

def construct_input_ref_pos_id_pair(input_ids, device):
	seq_length			= input_ids.size(1)
	position_ids		= torch.arange(seq_length, dtype=torch.long, device=device)
	ref_position_ids	= torch.zeros(seq_length, dtype=torch.long, device=device)

	position_ids		= position_ids.unsqueeze(0).expand_as(input_ids)
	ref_position_ids	= ref_position_ids.unsqueeze(0).expand_as(input_ids)
	return position_ids, ref_position_ids

def construct_input_ref_token_type_pair(input_ids, device):
	seq_len				= input_ids.size(1)
	token_type_ids		= torch.tensor([[0] * seq_len], dtype=torch.long, device=device)
	ref_token_type_ids	= torch.zeros_like(token_type_ids, dtype=torch.long, device=device)
	return token_type_ids, ref_token_type_ids

def construct_attention_mask(input_ids):
	return torch.ones_like(input_ids)

def get_word_embeddings(model):
	return model.distilbert.embeddings.word_embeddings.weight

def construct_word_embedding(model, input_ids):
	return model.distilbert.embeddings.word_embeddings(input_ids)

def construct_position_embedding(model, position_ids):
	return model.distilbert.embeddings.position_embeddings(position_ids)

def construct_type_embedding(model, type_ids):
	return model.distilbert.embeddings.token_type_embeddings(type_ids)

def construct_sub_embedding(model, input_ids, ref_input_ids, position_ids, ref_position_ids):
	input_embeddings				= construct_word_embedding(model, input_ids)
	ref_input_embeddings			= construct_word_embedding(model, ref_input_ids)
	input_position_embeddings		= construct_position_embedding(model, position_ids)
	ref_input_position_embeddings	= construct_position_embedding(model, ref_position_ids)
# 	input_type_embeddings			= construct_type_embedding(model, type_ids)
# 	ref_input_type_embeddings		= construct_type_embedding(model, ref_type_ids)

	return 	(input_embeddings, ref_input_embeddings), \
			(input_position_embeddings, ref_input_position_embeddings)

def get_base_token_emb(model, tokenizer, device):
	return construct_word_embedding(model, torch.tensor([tokenizer.pad_token_id], device=device))

def get_tokens(tokenizer, text_ids):
	return tokenizer.convert_ids_to_tokens(text_ids.squeeze())

def get_inputs(model, tokenizer, text, device):
	ref_token_id = tokenizer.mask_token_id
	sep_token_id = tokenizer.sep_token_id
	cls_token_id = tokenizer.cls_token_id

	input_ids, ref_input_ids		= construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id, device)
	position_ids, ref_position_ids	= construct_input_ref_pos_id_pair(input_ids, device)
# 	type_ids, ref_type_ids			= construct_input_ref_token_type_pair(input_ids, device)
	attention_mask					= construct_attention_mask(input_ids)

	(input_embed, ref_input_embed), (position_embed, ref_position_embed) = \
				construct_sub_embedding(model, input_ids, ref_input_ids, position_ids, ref_position_ids)

	return [input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, None, None, attention_mask]

# Create a wrapper function that captures the model
def create_forward_func(model):
    def forward_func(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=True):
        return nn_forward_func(model, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
    return forward_func

# attr_func = DiscretetizedIntegratedGradients(create_forward_func(model))


def calculate_attributions(model, tokenizer, inputs, device, attr_func, base_token_emb, nn_forward_func, get_tokens, target=None):
    # computes the attributions for given input

    # move inputs to main device
    inp = [x.to(device) if x is not None else None for x in inputs]

    # compute attribution
    scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
    # Pass target as scalar - Captum will handle the expansion internally
    attr, deltaa = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, 63, target=target)
    # compute metrics
    log_odd, pred	= calculate_log_odds(model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    comp			= calculate_comprehensiveness(model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    suff			= calculate_sufficiency(model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)

    #return log_odd
    return log_odd, comp, suff, attr, deltaa
