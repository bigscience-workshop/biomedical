from datasets import load_dataset
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel



def return_lm_baseline_zero_shot(model, tokenizer, template_with_mask):
    res = []
    inputs = tokenizer(template_with_mask, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 15, dim=1).indices[0].tolist()
    res= [tokenizer.decode([token]) for token in top_5_tokens]
    return res
    
    
    
def nli_based_zero_shot(model, tokenizer, premise, hypothesis):
    # run through model pre-trained on MNLI
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    return probs[:,1].item()