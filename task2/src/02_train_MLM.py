from datasets import load_dataset
from transformers import (BertTokenizer, DataCollatorForLanguageModeling,
                          AutoModelForMaskedLM, BertConfig,
                          Trainer, TrainingArguments)
from utils import load_config, seed_everything
import pandas as pd
import os

def prepare_datasets(tokenizer):

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['glycan'], padding='max_length', truncation=True, max_length=config['dataset']['max_length'])

    df_glycan = pd.read_pickle(config['paths']['df_glycan_path'])
    df_glycan['glycan'].to_csv(config['paths']['glycan_seqs'] , index=False)
    dataset = load_dataset('csv', data_files=config['paths']['glycan_seqs'])  

    # tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['glycan'])

    # Train-validation split
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=config['dataset']['val_size']) 
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    return train_dataset, val_dataset


def train_model(data_collator, train_dataset, val_dataset, tokenizer):
    
    # Initialize model
    model_config = BertConfig()
    model_config.vocab_size = tokenizer.vocab_size 
    model = AutoModelForMaskedLM.from_config(model_config)
    print(f'Initialized a model with {model.num_parameters()} parameters')

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=config['paths']['model'],
        overwrite_output_dir=True,
        eval_strategy = 'epoch',
        num_train_epochs=config['training']['n_epochs'],
        learning_rate=config['training']['lr'],
        weight_decay=config['training']['wd'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{config['paths']['model']}/logs",
        report_to='none'
    )

    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    print('START TRAINING')
    trainer.train()
    print(f"FINISH TRAINING. MODEL SAVED AT {config['paths']['model']}")
    
    return trainer


def save_logs(trainer):

    # Access the logged metrics
    log_history = trainer.state.log_history

    # Save train and validation loss to a file
    with open(f"{config['paths']['model']}/logs/loss_log.txt", "w") as f:
        for log in log_history:
            if "loss" in log or "eval_loss" in log:
                epoch = log.get("epoch", "N/A")
                train_loss = log.get("loss", "N/A")
                eval_loss = log.get("eval_loss", "N/A")
                f.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {eval_loss}\n")


if __name__ == '__main__':

    # for CUDA compatibility
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # for reproducibility
    seed_everything(42)

    # config file
    config = load_config("../configs/config.yaml")

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['paths']['tokenizer'])

    # get datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config['model']['mlm_probability']
    )

    # train and save the model
    trainer = train_model(data_collator, train_dataset, val_dataset, tokenizer)

    # save train/val loss per epoch in a file
    save_logs(trainer)
