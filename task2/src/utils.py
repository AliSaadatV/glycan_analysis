# Load YAML Configuration
def load_config(config_file):
    import yaml

    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    

# For reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_embed(glycan_seq, model, tokenizer):
    import torch

    # tokenize
    glycan_seq_tokens = tokenizer(glycan_seq, return_tensors="pt", padding=True, truncation=True)

    # forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**glycan_seq_tokens)

    # Extract hidden states
    last_hidden_state = outputs.last_hidden_state

    # calculate sequence embedding by averaging token embeddings
    attention_mask = glycan_seq_tokens['attention_mask']
    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
    mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

    return mean_embedding