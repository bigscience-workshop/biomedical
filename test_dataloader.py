from datasets import load_dataset

if __name__ == '__main__':
    data = load_dataset(f'biodatasets/bionlp_st_2019_bb', name="bionlp_st_2019_bb_kb+ner_source")
