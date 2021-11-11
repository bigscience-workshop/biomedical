
from datasets import load_dataset, interleave_datasets, logging
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd
from torch import nn
from collections import Counter
import torch
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM, BertModel, AutoModelWithLMHead, AutoModel, AutoTokenizer


# text = "Hey what are you up to man? Is everything alright?"
def truncate_text_for_prompt(text, tokenizer, allowed_max_length=500):
    inp = tokenizer(text)
    len_text = len(inp['input_ids'])
    if len_text > allowed_max_length:
        num_words_truncated = len_text-allowed_max_length
        allowed_start_idx = int(allowed_max_length/2)
        mask = inp['input_ids'][allowed_start_idx:allowed_start_idx+num_words_truncated]
        inp['input_ids'] = inp['input_ids'][:allowed_start_idx-1] + [tokenizer.sep_token_id] + inp['input_ids'][allowed_start_idx+num_words_truncated:] 
        inp['token_type_ids'] = inp['token_type_ids'][:allowed_start_idx-1] + [inp['token_type_ids'][0]] + inp['token_type_ids'][allowed_start_idx+num_words_truncated:] 
        inp['attention_mask'] = inp['attention_mask'][:allowed_start_idx-1] + [inp['attention_mask'][0]] + inp['attention_mask'][allowed_start_idx+num_words_truncated:] 
    else:
        inp = tokenizer(text, max_length=500, truncation=True, padding='max_length')
    return inp

def tokenizer_and_numericalize(example, truncate_text_for_prompt, tokenizer):
    sent = example['question'] + '[SEP]' + example['passage']
    inp = truncate_text_for_prompt(sent, tokenizer)
    example['input_ids'] =  inp['input_ids']
    example['token_type_ids'] =  inp['token_type_ids']
    example['attention_mask'] =  inp['attention_mask']
    example['label'] = int(example['answer'])
    return example


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def evaluate(dataloader, model, loss_fn, device):
    
    model.eval()
    epoch_losses_lm = []
    epoch_losses_tl = []
    epoch_accs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            task_labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            task_loss = loss_fn(outputs[0], task_labels)
            
            accuracy = get_accuracy(outputs[0], task_labels)
            epoch_losses_tl.append(task_loss.item())
            
            epoch_accs.append(accuracy.item())

    return epoch_losses_tl, epoch_accs