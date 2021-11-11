from datasets import load_dataset, interleave_datasets, logging
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd
from torch import nn
from collections import Counter
import torch
from transformers import AdamW


class TransformerForPrompting(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(self, ids, attn):
        # ids = [batch size, seq len]
        output = self.transformer(ids, attn, output_attentions=True, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:,0,:]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        prediction_vocab_size = self.transformer.cls(hidden)
        
        return prediction, prediction_vocab_size