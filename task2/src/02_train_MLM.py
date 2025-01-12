from datasets import load_dataset
from transformers import (BertTokenizer, DataCollatorForLanguageModeling,
                          RobertaConfig, RobertaForMaskedLM,
                          Trainer, TrainingArguments)
from utils import load_config, seed_everything
import pandas as pd


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
    train_test_split = tokenized_dataset['train'].train_test_split(test_size=config['dataset']['val_size'])  # 80% train, 20% validation
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    return train_dataset, val_dataset


def train_model(data_collator, train_dataset, val_dataset, tokenizer):
    # Set a configuration for our RoBERTa model
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=config['model']['max_position_embeddings'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        hidden_size=config['model']['hidden_size'],
        type_vocab_size=config['model']['type_vocab_size']
    )

    # Initialize the model from a configuration without pretrained weights
    model = RobertaForMaskedLM(config=config)
    print(f'Initialized a model with {model.num_parameters()} parameters')

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=config['paths']['model'],
        overwrite_output_dir=True,
        evaluation_strategy = 'epoch',
        num_train_epochs=config['training']['batcn_epochs'],
        learning_rate=config['training']['lr'],
        weight_decay=config['training']['wd'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        save_strategy='best',
        logging_dir=f"{config['paths']['model']}/logs"
    )

    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print('START TRAINING')
    # Train the model
    trainer.train()
    print(f"FINISH TRAINING. MODEL SAVED AT {config['paths']['model']}")


if __name__ == '__main__':
    # for reproducibility
    seed_everything(42)

    # config file
    config = load_config("../configs/config.yaml")

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('../results/misc/tokenizer/')

    # get datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config['model']['mlm_probability']
    )

    # train and save the model
    train_model(data_collator, train_dataset, val_dataset, tokenizer)