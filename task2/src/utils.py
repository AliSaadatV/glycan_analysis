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


def train_classifier(
    df, 
    feature_cols, 
    label_cols, 
    multi_label=False, 
    max_iter=1000, 
    n_jobs=-1, 
    normalize=False, 
    scaler_type="standard", 
    use_class_weight=True
):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Extract features and labels, drop NaNs
    X = df.dropna()[feature_cols]
    y = df.dropna()[label_cols]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select scaler based on the user input
    if normalize:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
    else:
        scaler = None

    # Initialize the Logistic Regression model
    if multi_label:
        base_model = LogisticRegression(solver='saga', max_iter=max_iter, n_jobs=n_jobs)
        model = MultiOutputClassifier(base_model)
    else:
        if use_class_weight:
            # Compute class weights for single-label classification
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            base_model = LogisticRegression(solver='saga', max_iter=max_iter, n_jobs=n_jobs, class_weight=class_weight_dict)
        else:
            base_model = LogisticRegression(solver='saga', max_iter=max_iter, n_jobs=n_jobs)
        model = base_model

    # Create pipeline with optional scaling
    if normalize:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', model)
        ])
    else:
        pipeline = model

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = pipeline.predict(X_test)

    # Evaluate metrics
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_all_labels = f1_score(y_test, y_test_pred, average=None)
    accuracy = accuracy_score(y_test, y_test_pred)

    return pipeline, f1_macro, f1_all_labels, accuracy
