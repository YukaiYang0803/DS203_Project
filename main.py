#  Import packages
import pandas as pd
from utils import generate_query,gpt3,gpt2,add_response,add_response_gpt2,score,score_gpt2,save_as_json
import os
import random
import argparse
from IPython.display import clear_output
import json
import time

#  Add arguments
parser = argparse.ArgumentParser(description='Generate the query and run the experiment.')

parser.add_argument('API_key',
                    default = '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',
                    type=str,
                    help='API key for calling GPT-3')
parser.add_argument('mode', 
                    default=1,
                    type=int,
                    help='Choose the new words generate mode')
parser.add_argument('model', 
                    default='gpt3',
                    type=str,
                    help='Choose the model to evaluate')
args = parser.parse_args()

## Get arguments
#  Expose the API_KEY to the system
api = args.API_key
os.environ['OPENAI_API_KEY'] = api
#  determine the mode
mode = args.mode
model_str = args.model
if mode == 1:
    mode = '_typo'
elif mode == 2:
    mode = '_prefix_suffix'
elif mode == 3:
    mode = '_replace_pre_suffix'
elif mode == 4:
    mode = '_random'
# data_dir = args.data_dir[:-4] # only want the name so remove ".csv"

if model_str == 'gpt3':
    get_response = add_response
else: # model_str == 'gpt2'
    get_response = add_response_gpt2

#  Read the files
file = f'data{mode}.csv'
df = pd.read_csv(file)

'''
# Enter the query to run the task
query1 = "The instructions were not just qwerasting, they were positively misleading.\nWhich of the following explains the word qwerasting in the sentence better\n(A) hard to understand\n(B) unhappy\n(C) clear\n(D) angry?\nProvide the answer in A, B, C, or D.\n "
#response1 = gpt3(query1)
#print(response1)

query2 = "What is the capital city of China? Answer in one word.\n"


queries = {1:query1,2:query2}
choices = {1:'A',2:'Beijing'}
y_pred = {}

for i in queries.keys():
    query = queries[i]
    y_pred[i] = add_response(query)
'''

#  Prepare to store the experiment data and result
random_state = 2022
queries = {}
y_pred,y_true = {},{}
egs = """
Shall we forever punctusign the pleasure of construction to the carpenter?
I am in no hurry to punctusign my office and be planted, you may be sure.
The Capitulation of Wittenberg (1547) is the name given to the treaty by which John Frederick the Magnanimous was compelled to punctusign the electoral dignity and most of his territory to the Albertine branch of the Saxon family.
Which of the following explains the word punctusign in the sentence better
A. throw
B. retire
C. resign
D. continue
Provide the answer in A, B, C, or D.
C
"""


#  Generate the query
for i in range(len(df)):
    row = df.iloc[i]
    query,y = generate_query(row,i,mode)
    if query == None:
        break
        
    queries[i] = query
    y_true[i] = y

#random.seed(random_state)
# this one shall not be a fixed random (always shuffle the query randomly)
keys = list(queries.keys())
random.shuffle(keys)

#  Submit all queries
for i in keys:#range(len(queries)):
    #query = queries[i] # in query order
    query = queries[i] # in random order
    print(f'Requesting query {i}...')
    print(query)
    if model_str == 'gpt3':
        y_pred[i] = get_response(egs+query)
    else:
        y_pred[i] = get_response(query,model_str)
    #clear_output()
    time.sleep(1)
print('All queries done!')

# match answer (ABCD) with index numbers (0123) for metrics (acc) computation later
if model_str == 'gpt3':
    mapping = {0:'A',1:'B',2:'C',3:'D'}
    y_true = dict(map(lambda item: (item[0], mapping[item[1]]), y_true.items()))
    y_pred = dict(map(lambda item: (item[0], item[1][0]), y_pred.items()))
    acc = score(y_true,y_pred)
else: # model_str == 'gpt2'
    y_true = dict(map(lambda item: (item[0], mapping[item[1]]), y_true.items()))
    #y_pred = dict(map(lambda item: (item[0], item), y_pred.items()))
    acc = score_gpt2(y_true,y_pred)

#print(choices,y_pred)

print(acc)

save_as_json(y_true,f'y_true{mode}')
save_as_json(y_pred,f'y_pred{mode}')
