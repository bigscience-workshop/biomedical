from datasets import load_dataset
from bigbio.dataloader import BigBioConfigHelpers

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from umap_reducer import UMAPReducer
import pandas as pd


def main():
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    data = load_dataset("bigbio/biodatasets/chemdner/chemdner.py", name="chemdner_bigbio_text")

    sample_size = 50
    sample_idc = np.random.randint(0, len(data['train'])-1, size=sample_size)
    sentences = []
    for i in sample_idc:
        sentences.append(data['train'][int(i)]['text'])
        # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)


    reducer = UMAPReducer()
    umap_embeddings = reducer.embed(sentence_embeddings)
    print(umap_embeddings.shape)
    clusters = reducer.clusterAnalysis(umap_embeddings)
    print(clusters.labels_.tolist())
    c_labels = clusters.labels_.tolist()

    df_data = []
    for sent, coords, label in zip(sentences, umap_embeddings, c_labels):
        row = {}
        row['sent'] = sent
        row['x'] = coords[0]
        row['y'] = coords[1]
        row['z'] = coords[2]
        row['label'] = label
        df_data.append(row)

    df = pd.DataFrame(df_data)
    #df sent, x, y, z, label
    print(df.head())
    return df


if __name__ == '__main__':
    main()
