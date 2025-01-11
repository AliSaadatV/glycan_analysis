import pandas as pd
from utils import load_config
from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import os

def column_to_files(column, prefix, txt_files_dir):
    # The prefix is a unique ID to avoid to overwrite a text file
    i=prefix
    #For every value in the df, with just one column
    for row in column.to_list():
      # Create the filename using the prefix ID
      file_name = os.path.join(txt_files_dir, str(i)+'.txt')
      try:
        # Create the file and write the column text to it
        f = open(file_name, 'wb')
        f.write(row.encode('utf-8'))
        f.close()
      except Exception as e:  #catch exceptions(for eg. empty rows)
        print(row, e) 
      i+=1
    # Return the last ID
    return i


if __name__ == '__main__':
    # config file
    config = load_config("../configs/config.yaml")
    # df that contains glycans sequences
    df_glycan = pd.read_pickle(config['paths']['df_glycan_path'])

    # Prepare files for training
    data = df_glycan["glycan"]
    # Removing the end of line character \n
    data = data.replace("\n"," ")
    # Set the ID to 0
    prefix=0
    # Create a file for every description value
    prefix = column_to_files(data, prefix, '../results/misc/train_tokenizer_input/')

    # List of file paths
    paths = [str(x) for x in Path("../results/misc/").glob("train_tokenizer_input/*.txt")]
    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(lowercase=True)
    # Customize training
    tokenizer.train(files=paths,
                    vocab_size=config['tokenizer']['vocab_size'],
                    min_frequency=config['tokenizer']['min_frequency'],
                    show_progress=True)
    #Save the Tokenizer to disk
    tokenizer.save_model(config['paths']['tokenizer'])
