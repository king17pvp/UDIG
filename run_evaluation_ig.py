import json
import math
import time
import tqdm
import torch
import random
import inspect
import argparse
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Literal, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, ShapleyValueSampling, Lime
from sklearn.neighbors import kneighbors_graph
from datasets import load_dataset
from dig import DiscretetizedIntegratedGradients
from attributions import run_dig_explanation
from metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency
from captum.attr._utils.common import _reshape_and_sum, _validate_input
from sklearn.neighbors import kneighbors_graph
from xai_metrics import *
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cache = {}
cache_ = {}
def integrated_gradient(
    text: str,
    steps: int,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,  # set True if you want [CLS]/[SEP] included in the output
):
    """
    Rienman-sum of integrated gradient
    Returns: dict(token, attributions) with attributions of shape (L,) (include specials)
    """
    global cache
    if cache.get(model_name, None) is None:
        tmp = {}
        print(f"Model {model_name} not found in cache, loading from stratch")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tmp["model"] = model
        tmp["tokenizer"] = tokenizer
        cache[model_name] = tmp
    else:
        tokenizer = cache[model_name]["tokenizer"]
        model = cache[model_name]["model"]
    model.eval()
    model.zero_grad()
    enc = tokenizer(text, return_tensors="pt", truncation=True, return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"].to(device)           # (1, L)
    attention_mask = enc["attention_mask"].to(device) # (1, L)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # Only pass token_type_ids if the model forward() accepts it
    fwd_params = inspect.signature(model.forward).parameters
    extra_kwargs = {}
    if "token_type_ids" in fwd_params and token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    # Base embeddings X: (1, L, d)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d)
    L, d = X.shape[1], X.shape[2]

    if tokenizer.pad_token_id is not None:
        with torch.no_grad():
            X0 = embed(torch.full_like(input_ids, tokenizer.pad_token_id))
    else:
        X0 = torch.zeros_like(X)
    with torch.no_grad():
        final_logits = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs).logits[0]
    pred_id = int(final_logits.argmax(dim=-1).item())
    # Vector from baseline to input
    start_time = time.perf_counter()
    delta = (X - X0)  # (1, L, d)
    t_vals =  torch.linspace(1.0 / steps, 1.0, steps, device=device, dtype=X.dtype)
    total_grad = torch.zeros_like(X)  # (1, L, d)
    for t in t_vals:
        X_t = (X0 + t * delta).detach().requires_grad_(True)  # (1, L, d)

        # Forward pass using inputs_embeds
        out = model(inputs_embeds=X_t, attention_mask=attention_mask, **extra_kwargs)
        logits = out.logits  # (1, C)

        # Select the scalar output for the target class
        # Using raw logit is standard for IG;
        target = logits[0, pred_id]
        # Backprop to get d target / d X_alpha
        grads = torch.autograd.grad(
            outputs=target, inputs=X_t, retain_graph=False, create_graph=False
        )[0]  # (1, L, d)
        total_grad += grads

    avg_grad = total_grad / steps  # (1, L, d)

    # Elementwise multiply by the input difference (path length) and sum over embedding dim
    attributions = (delta * avg_grad).sum(dim=-1).squeeze(0).detach().cpu()  # (L,)
    end_time = time.perf_counter()
    # Also compute the final logits/prediction for reference
    

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    position_emb = construct_position_embedding(model, position_ids)
    inp = get_inputs(model, tokenizer, text, device)
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
    log_odd, pred = calculate_log_odds(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attributions, topk=20)
    comp = calculate_comprehensiveness(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attributions, topk=20)
    suff = calculate_sufficiency(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attributions, topk=20)
    if not show_special_tokens:
        # Typical HuggingFace BERT tokenization has specials at positions 0 and last
        # We detect them via tokenizer’s special tokens set.
        special_ids = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
        tokens = [tokens[i] for i in keep_idx]
        attributions = attributions[keep_idx]
    return {
        "tokens": tokens,
        "attributions": attributions,        # shape (L,)
        "logits": final_logits.detach().cpu(),  # shape (C,)
        "log_odd": log_odd,
        "comp": comp,
        "suff": suff,
        "time": end_time - start_time,
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert', help='Model name or path')

    args = parser.parse_args()
    model = args.model
    if model == 'distilbert':
        from distilbert_helper import *
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    elif model == 'bert':
        from bert_helper import *
        model_name = "textattack/bert-base-uncased-SST-2"
    elif model == 'roberta':
        from roberta_helper import *
        model_name = "textattack/roberta-base-SST-2"
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    dataset= load_dataset('glue', 'sst2')['test']
    data= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
    # dataset	= load_dataset('imdb')['test']
    # data = list(zip(dataset['text'], dataset['label']))
    steps = 500
    count = 0
    # compute the DIG attributions for all the inputs
    print('Starting attribution computation...')
    inputs,delta_pcs_list = [],[]
    log_odds, comps, suffs, deltas, delta_pcs, count, total_time = 0, 0, 0, 0, 0, 0, 0
    print_step = 100
    for row in tqdm(data):
        text = row[0]
        # res_egrad = egrad_integral_bert(text = text, a = a, b = b, steps = steps, show_special_tokens = False)
        res_intergrated_grad = integrated_gradient(text = text, steps = steps, model_name = model_name, show_special_tokens = False)
        # res_shapley = shapley_bert(text = text, show_special_tokens = False)
        # res_lime = lime_bert(text = text, use_pad_baseline=True, show_special_tokens = False)
        # res_udig = udig_bert(text = text, show_special_tokens = False)
        # print(f"Result for UDIG, took {res_udig['time']} seconds:")
        # print(f"Log odd: {res_udig['log_odd']}, Comprehensiveness: {res_udig['comp']}, Sufficiency: {res_udig['suff']}")
        # print(f"Result for IG, took {res_intergrated_grad['time']} seconds: ")
        # print(f"Log odd: {res_intergrated_grad['log_odd']}, Comprehensiveness: {res_intergrated_grad['comp']}, Sufficiency: {res_intergrated_grad['suff']}")
        # print(f"Result for Shapley, took {res_shapley['time']} seconds: ")
        # print(f"Log odd: {res_shapley['log_odd']}, Comprehensiveness: {res_shapley['comp']}, Sufficiency: {res_shapley['suff']}")
        # print(f"Result for LIME, took {res_lime['time']} seconds: ") 
        # print(f"Log odd: {res_lime['log_odd']}, Comprehensiveness: {res_lime['comp']}, Sufficiency: {res_lime['suff']}")
        # print(f"Result for Epsilon Gradient, took {res_egrad['time']} seconds:")
        # print(f"Log odd: {res_egrad['log_odd']}, Comprehensiveness: {res_egrad['comp']}, Sufficiency: {res_egrad['suff']}")
        log_odds += res_intergrated_grad['log_odd']
        comps += res_intergrated_grad['comp']
        suffs += res_intergrated_grad['suff']
        total_time += res_intergrated_grad['time']
        count += 1
        if count % print_step == 0:
            print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
                'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))
            # print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
            #       'Sufficiency: ', np.round(suffs / count, 4),  'Avg delta: ', np.round(deltas / count, 4))

    print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
        'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))