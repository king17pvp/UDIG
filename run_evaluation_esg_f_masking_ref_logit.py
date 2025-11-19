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
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
cache = {}
cache_ = {}

def power_renorm(x, p=10):
    """ L_p normalization """
    x_p = x.abs().pow(p)
    x_p_sum = x_p.sum()
    x_p_norm = x_p / (x_p_sum + 1e-5)
    return x_p_norm

def esg_f_masking_ref_logit_bert_db(
    text: str,
    a: float = -1.0,
    b: float = 1.0,
    steps: int = 101,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_margin_if_binary: bool = True,   # sigmoid(logit[label]-logit[other]) when num_labels==2,
    show_special_tokens: bool = False,
):
    """
    Riemann-sum integral of ε-gradient along ε(t) = t * 1 for t in [a, b],
    Returns: dict(tokens, attributions) with attributions of shape (L,) (includes specials).
    """
    # --- Load model/tokenizer ---
    global cache
    if cache.get(model_name, None) is None:
        print(f"Model {model_name} not found in cache, loading from stratch")
        tmp = {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tmp["model"] = model
        tmp["tokenizer"] = tokenizer
        cache[model_name] = tmp
    else:
        tokenizer = cache[model_name]["tokenizer"]
        model = cache[model_name]["model"]
    model.to(device)
    model.eval()

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
        final_logits = model(inputs_embeds=X, attention_mask=attention_mask).logits[0]
        
        # print(f"Final logits: {final_logits}")
    pred_id = int(final_logits.argmax(dim=-1).item())
    L, d = X.shape[1], X.shape[2]
    start_time = time.perf_counter()
    

    # Integration grid
    t_vals = torch.linspace(a, b, steps, device=device, dtype=X.dtype)
    dt = float((b - a) / max(1, steps - 1)) if steps > 1 else float(b - a)
    # Accumulator
    attr = torch.zeros(L, device=device, dtype=X.dtype)
    ones_L = torch.ones(L, device=device, dtype=X.dtype)
    direct = X.squeeze()
    #direct = direct + torch.randn(direct.shape).to(device)/1000
    pre_score = None
    target_score = None
    EXPLORATION = 1000 # Higher for more exploration
    attrs = []
    db_scores = []
    # print(X.shape)
    # print(t_vals)
    
    out = model(
        inputs_embeds=X,
        attention_mask=attention_mask,
        **extra_kwargs
    )
    logits = out.logits[0]  # (num_labels,)
    probs = F.softmax(logits, dim=-1)
    target_prelogit = logits[pred_id]
    target_prob = probs[pred_id]
    ts = []
    n_base = L*3
    ps = 0.9
    p_sample = torch.full((L,1), ps).to(device)
    prev_lb_score = None
    prev_lg_score = None
    mask_token = tokenizer.mask_token            # usually "[MASK]"
    mask_token_id = tokenizer.mask_token_id      # e.g. 103 for BERT

    mask_token_tensor = torch.tensor([[mask_token_id]], device=device)
    mask_embedding = model.get_input_embeddings()(mask_token_tensor.clone().contiguous())
    X_RefMask = mask_embedding.repeat(1,L,1)
    def func(x):
        iterpolated = x.tile((1,d))
        padding_mask = torch.full((L,1), 1).to(device)
        padding_mask[0] = 0
        padding_mask[-1] = 0

        iterpolated[0,:] = 1
        iterpolated[-1,:] = 1


        #X_Ref = X * masking_w
        X_Ref = X_RefMask

        X_inter = X * iterpolated  + X_Ref * (1-iterpolated)
        eps = torch.zeros((1,L,1), device=device, dtype=X.dtype).requires_grad_(True)
        #eps = torch.zeros(X_inter.shape, device=device, dtype=X.dtype).requires_grad_(True)
        X_inter = X_inter + eps * padding_mask
    
        out = model(
            inputs_embeds=X_inter,
            attention_mask=attention_mask,
            **extra_kwargs
        )
        logits = out.logits[0]  # (num_labels,)

        probs = F.softmax(logits, dim=-1)  # convert logits -> probability distribution
        # score = probs[label_id]
        logit_score = logits[pred_id]
        label_score = probs[pred_id]
        return logit_score
    start_time = time.perf_counter()
    for m in range(1):
        sum_dlg = 0
        for i in range(len(t_vals)):
            # ε(t) = t * 1_L  -> (L,)
            #ones_L_rand = torch.randn(L).to(device)
            ones_L_rand = torch.ones(L).to(device)
            #ones_L_rand = ones_L_rand * (b-a) + a
            t = t_vals[i]
            ts.append(t.item())
    
            inteprolate_v = t * ones_L_rand
            #inteprolate_v[-5] = 0
            #inteprolate_v[-2] = 0
            itepolated_o = inteprolate_v.view(L,1)
            ex = torch.zeros((L,1), device=device, dtype=X.dtype).requires_grad_(True)
            itepolated_o = itepolated_o + ex
            iterpolated = itepolated_o.tile((1,d))


      
            padding_mask = torch.full((L,1), 1).to(device)
            padding_mask[0] = 0
            padding_mask[-1] = 0

            iterpolated[0,:] = 1
            iterpolated[-1,:] = 1


            #X_Ref = X * masking_w
            X_Ref = X_RefMask

            X_inter = X * iterpolated  + X_Ref * (1-iterpolated)
            eps = torch.zeros((1,L,1), device=device, dtype=X.dtype).requires_grad_(True)
            #eps = torch.zeros(X_inter.shape, device=device, dtype=X.dtype).requires_grad_(True)
            X_inter = X_inter + eps * padding_mask
        
            out = model(
                inputs_embeds=X_inter,
                attention_mask=attention_mask,
                **extra_kwargs
            )
            logits = out.logits[0]  # (num_labels,)

            probs = F.softmax(logits, dim=-1)  # convert logits -> probability distribution
            # score = probs[label_id]
            logit_score = logits[pred_id]
            label_score = probs[pred_id]
            dscore = target_prob - label_score
            if i == 0:
                prev_label_score = label_score
                prev_lg_score = logit_score
            dlogit = logit_score - prev_lg_score    
            dlb = label_score - prev_label_score
            prev_label_score = label_score
            prev_lg_score = logit_score
            # ∂score/∂ε  (L,)
            (grad_eps,) = torch.autograd.grad(logit_score, itepolated_o, retain_graph=False, create_graph=False)
            grad_eps_n = grad_eps
            grad_eps_n = grad_eps / (torch.sum(grad_eps) + 1e-10)
            #print(torch.sum(grad_eps_n))
            grad_eps_n = grad_eps_n.squeeze() #* (masking_w.squeeze() <0) #* itepolated_o.squeeze()
            #grad_eps_n = grad_eps_n / (torch.sum(torch.abs(grad_eps_n)) + 1e-10)

            db_scores.append((label_score.detach().cpu().numpy(), logit_score.detach().cpu().numpy(), dlb.detach().cpu().numpy(), t.item()))
            # accumulate ∫ grad_ε · dε, with dε = 1_L * dt
            attri = grad_eps_n * dlogit #* dscore # * lb_score
            sum_dlg += dlb
            attrs.append(attri)
            attr += attri
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # print("Sum dlb:", sum_dlg)
    end_time = time.perf_counter()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, text, device)
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
    log_odd, pred = calculate_log_odds(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    comp = calculate_comprehensiveness(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    suff = calculate_sufficiency(nn_forward_func, model, X, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20)
    if not show_special_tokens:
        # Typical HuggingFace BERT tokenization has specials at positions 0 and last
        # We detect them via tokenizer’s special tokens set.
        special_ids = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
        tokens = [tokens[i] for i in keep_idx]
        attr = attr[keep_idx]
    return {
        "tokens": tokens, 
        "attributions": attr.detach().cpu(), 
        "attributions_steps": attrs, 
        "db_scores": db_scores, 
        "ts": ts,
        "time": end_time - start_time,
        "log_odd": log_odd,
        "comp": comp,
        "suff": suff
    }  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert', help='Model name or path')
    parser.add_argument('--dataset', choices=['sst2', 'imdb', 'rotten'])
    parser.add_argument('--range', type=float, nargs=2, default=[-1.0, 1.0], help='Range [a,b] for epsilon')
    parser.add_argument('--steps', type=int, default=101, help='Number of steps for Riemann sum')

    args = parser.parse_args()
    a, b = args.range
    steps = args.steps
    model = args.model
    dataset_name = args.dataset
    if model == 'distilbert':
        from distilbert_helper import *
        if dataset_name == 'sst2':
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        if dataset_name == 'imdb':
            model_name = "textattack/distilbert-base-uncased-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/distilbert-base-uncased-rotten-tomatoes"
    elif model == 'bert':
        from bert_helper import *
        if dataset_name == 'sst2':
            model_name = "textattack/bert-base-uncased-SST-2"
        elif dataset_name == 'imdb':
            model_name = "textattack/bert-base-uncased-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/bert-base-uncased-rotten-tomatoes"
    elif model == 'roberta':
        from roberta_helper import *
        if dataset_name == 'sst2':
            model_name = "textattack/roberta-base-SST-2"
        elif dataset_name == 'imdb':
            model_name = "textattack/roberta-base-imdb"
        elif dataset_name == 'rotten':
            model_name = "textattack/roberta-base-rotten-tomatoes"
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using range [{a}, {b}] for epsilon with {steps} steps")
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res_egrad = esg_f_masking_ref_logit_bert_db(text, a = a, b = b, steps = steps, model_name = model_name, show_special_tokens = False)
    print("This is Epsilon gradient results:")
    print(f"Running took {res_egrad['time']} seconds")
    print(f"Log odd: {res_egrad['log_odd']}, Comprehensiveness: {res_egrad['comp']}, Sufficiency: {res_egrad['suff']}")
    for tok, val in zip(res_egrad["tokens"], res_egrad["attributions"]):
        print(f"{tok:>12s} : {val.detach().cpu().numpy():+.6f}")
    if args.dataset == 'imdb':
        dataset	= load_dataset('imdb')['test']
        data	= list(zip(dataset['text'], dataset['label']))
        data	= random.sample(data, 2000)
    elif args.dataset == 'sst2':
        dataset	= load_dataset('glue', 'sst2')['test']
        data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
    elif args.dataset == 'rotten':
        dataset	= load_dataset('rotten_tomatoes')['test']
        data	= list(zip(dataset['text'], dataset['label']))
    # dataset	= load_dataset('imdb')['test']
    # data = list(zip(dataset['text'], dataset['label']))
    count = 0
    # compute the DIG attributions for all the inputs
    print('Starting attribution computation...')
    inputs,delta_pcs_list = [],[]
    log_odds, comps, suffs, deltas, delta_pcs, count, total_time = 0, 0, 0, 0, 0, 0, 0
    print_step = 100
    for row in tqdm(data):
        text = row[0]
        res_egrad = esg_f_masking_ref_logit_bert_db(text = text, a = a, b = b, steps = steps, model_name = model_name, show_special_tokens = False)
        log_odds += res_egrad['log_odd']
        comps += res_egrad['comp']
        suffs += res_egrad['suff']
        total_time += res_egrad['time']
        count += 1
        if count % print_step == 0:
            print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
                'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))
            # print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
            #       'Sufficiency: ', np.round(suffs / count, 4),  'Avg delta: ', np.round(deltas / count, 4))

    print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
        'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))