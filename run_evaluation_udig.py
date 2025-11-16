import json
import math
import time
import tqdm
import torch
import random
import argparse
import inspect
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
def udig_bert( 
    text,  
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), 
    strategy='maxcount',
    steps=30,
    nbrs=50,
    factor=1,
    show_special_tokens = False
):
    global cache_
    if cache_.get(model_name, None) is None:
        print(f"Model {model_name} not found in cache, loading from stratch")
        tmp = {}
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        model.eval()
        model.zero_grad()
        word_features        = get_word_embeddings(model).cpu().detach().numpy()
        word_idx_map        = tokenizer.vocab
        A                    = kneighbors_graph(word_features, 500, mode='distance', n_jobs=-1)
        auxiliary_data = [word_idx_map, word_features, A]
        attr_func = DiscretetizedIntegratedGradients(create_forward_func(model))
        tmp["model"] = model
        tmp["tokenizer"] = tokenizer
        tmp["auxiliary_data"] = auxiliary_data
        tmp["attr_func"] = attr_func
        cache_[model_name] = tmp
    else:
        model = cache_[model_name]["model"]
        tokenizer = cache_[model_name]["tokenizer"]
        auxiliary_data = cache_[model_name]["auxiliary_data"]
        attr_func = cache_[model_name]["attr_func"]
    base_token_emb = get_base_token_emb(model, tokenizer, device)
    inp = get_inputs(model, tokenizer, text, device)
    start_time = time.perf_counter()
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
    scaled_features         = monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
                                        device, auxiliary_data, method ="UIG", steps=steps, nbrs = nbrs, factor=factor, strategy=strategy)
    inputs                    = [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]
    with torch.no_grad():
        final_logits = model(inputs_embeds=input_embed, attention_mask=attention_mask).logits[0]
    pred_id = int(final_logits.argmax(dim=-1).item())
    target_label = pred_id  
    log_odd, comp, suff, attrib, delta= calculate_attributions(nn_forward_func, model, tokenizer, inputs, device, attr_func, base_token_emb, nn_forward_func, get_tokens, target=target_label)
    end_time = time.perf_counter()
    tokens = get_tokens(tokenizer, input_ids)
    if not show_special_tokens:
        # Typical HuggingFace BERT tokenization has specials at positions 0 and last
        # We detect them via tokenizer’s special tokens set.
        special_ids = set(tokenizer.all_special_ids)
        keep_idx = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
        tokens = [tokens[i] for i in keep_idx]
        attrib = attrib[keep_idx]
    return {
        "tokens": tokens,
        "attributions": attrib.detach().cpu(),
        "delta": delta,
        "log_odd": log_odd,
        "comp": comp,
        "suff": suff,
        "time": end_time - start_time
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
    text = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    res_udig = udig_bert(text, model_name = model_name, show_special_tokens = False)
    print(f"Running took {res_udig['time']} seconds")
    print(f"Log odd: {res_udig['log_odd']}, Comprehensiveness: {res_udig['comp']}, Sufficiency: {res_udig['suff']}")
    for tok, val in zip(res_udig["tokens"], res_udig["attributions"]):
        print(f"{tok:>12s} : {val.detach().cpu().numpy():+.6f}")
    count = 0
    # compute the DIG attributions for all the inputs
    print('Starting attribution computation...')
    inputs,delta_pcs_list = [],[]
    log_odds, comps, suffs, deltas, delta_pcs, count, total_time = 0, 0, 0, 0, 0, 0, 0
    print_step = 100
    for row in tqdm(data):
        text = row[0]
        # res_egrad = egrad_integral_bert(text = text, a = a, b = b, steps = steps, show_special_tokens = False)
        # res_intergrated_grad = integrated_gradient(text = text, steps = steps, show_special_tokens = False)
        # res_shapley = shapley_bert(text = text, show_special_tokens = False)
        # res_lime = lime_bert(text = text, use_pad_baseline=True, show_special_tokens = False)
        res_udig = udig_bert(text = text, model_name = model_name, show_special_tokens = False)
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
        log_odds += res_udig['log_odd']
        comps += res_udig['comp']
        suffs += res_udig['suff']
        total_time += res_udig['time']
        count += 1
        if count % print_step == 0:
            print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
                'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))
            # print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
            #       'Sufficiency: ', np.round(suffs / count, 4),  'Avg delta: ', np.round(deltas / count, 4))

    print('Log-odds: ', np.round(log_odds / count, 4), 'Comprehensiveness: ', np.round(comps / count, 4), 
        'Sufficiency: ', np.round(suffs / count, 4), "Time: ", np.round(total_time / count, 4))