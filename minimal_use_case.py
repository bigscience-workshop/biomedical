import os
import unittest
from dataloader import BioDataset

# Simplest use case
# Will download data to currdir 
# Args: <dataset_name>, <data_root>, 
dataset = BioDataset("ddi")

# Example: Get document 3 (python index 2) from the train
doc = dataset.data.train[2]

# Show the text (in form str)
print(doc.text)

# Show the entities (in form List[Entity])
print(doc.entities)

# Show the relations (in form List[Relation])
print(doc.relations)
