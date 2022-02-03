# Based on predictor format of Sci-Tail

# this is the BARE MINIMUM; it can be super-setted into the overall schema.
features = datasets.Features(
    {
        "question": datasets.Value("string"),
        "text1": datasets.Value("string"),  # Passage/sentence 1
        "text2": datasets.Value("string"),  # Passage/sentence 
        "label": datasets.Value("string"),  # entails/etc
    }
)

## We may want to consider cases where metadata (i.e. "number of volunteers that annotated this answer") etc. for gold labeling - I also couldn't find something but happy to keep a hierarchical structure