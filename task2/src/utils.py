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


def train_classifier(df, feature_cols, label_cols, multi_label=False, max_iter=1000, n_jobs=-1):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import f1_score, accuracy_score

    X = df.dropna()[feature_cols]
    y = df.dropna()[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='saga', max_iter=max_iter, n_jobs=n_jobs)

    if multi_label:
        model = MultiOutputClassifier(model)

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_all_labels = f1_score(y_test, y_test_pred, average=None)
    accuracy = accuracy_score(y_test, y_test_pred)

    return model, f1_macro, f1_all_labels, accuracy