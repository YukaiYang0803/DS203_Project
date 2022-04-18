#  Import packages
import pandas as pd
import random
import argparse
from utils import change_word

#  Add arguments
#arg_default = set({'data_dir':'data.csv','mode':'1'})
parser = argparse.ArgumentParser(description='Change the word to "new_word".')
parser.add_argument('data_dir', 
                    default = 'data.csv',
                    type=str,
                    help='Name of the data file')
parser.add_argument('mode', 
                    default=1,
                    type=int,
                    help='Choose a mode to generate new words')
args = parser.parse_args()

# Get arguments
mode = args.mode
file = args.data_dir


if mode == 1:
    new_col = '_typo'
'''
if mode == 2:
    new_col = '_prefix_suffix'''

#  read the file
df = pd.read_csv(file)
df['word'] = df['word'].str.lower() # lower all original words for str.find() method later


## Change word to typo
#  First add the columns for the sides
df['mask'+new_col] = None
df['sen1'+new_col],df['sen2'+new_col],df['sen3'+new_col] = None,None,None

## Add the new words and sentences

# avoid warning when changing elements in the df
pd.options.mode.chained_assignment = None  # default='warn'

# generate new words
for (idx, word) in enumerate(df['word']):
    df['mask_typo'][idx] = change_word(word,idx)

new_word_col = 'mask'+new_col

for idx in range(len(df)):
    word = df['word'][idx]
    word_length = len(word)
    cols = ['sen1','sen2','sen3']
    for col in cols:
        if pd.isnull(df[col][idx]):
            break
        word_ind = df[col][idx].lower().find(word)
        word_ind = [word_ind, word_ind+word_length]
        new_word = df[new_word_col][idx]
        df[col+'_typo'][idx] = df[col][idx][:word_ind[0]]+new_word+df[col][idx][word_ind[1]:]


df.to_csv(f'data{new_col}.csv')
