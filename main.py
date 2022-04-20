#  Import packages
import pandas as pd
from utils import generate_query,gpt3,add_response,score,save_as_json
import os
import random
import argparse
from IPython.display import clear_output
import json


#  Add arguments
parser = argparse.ArgumentParser(description='Generate the query and run the experiment.')

parser.add_argument('API_key',
                    default = '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',
                    type=str,
                    help='API key for calling GPT-3')
'''
parser.add_argument('datafile_name', 
                    default='data.csv',
                    type=str,
                    help='name of the query file (currently must be in the same dir)')
'''
parser.add_argument('mode', 
                    default=1,
                    type=int,
                    help='Choose the new words generate mode')
args = parser.parse_args()

## Get arguments
#  Expose the API_KEY to the system
api = args.API_key
os.environ['OPENAI_API_KEY'] = api
#  determine the mode
mode = args.mode
if mode == 1:
    mode = '_typo'
# data_dir = args.data_dir[:-4] # only want the name so take off ".csv"

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

#  Generate the query
for i in range(len(df)):
    row = df.iloc[i]
    query,y = generate_query(row,i)
    
    if query == None:
        break
        
    queries[i] = query
    y_true[i] = y

random.seed(random_state)
keys = list(queries.keys())
random.shuffle(keys)
#  Submit all queries
for i in keys:#range(len(queries)):
    #query = queries[i] # in query order
    query = queries[i] # in random order
    print(f'Requesting query {i}...')
    print(query)
    y_pred[i] = add_response(query)
    clear_output()
print('All queries done!')
    
mapping = {0:'A',1:'B',2:'C',3:'D'}
y_true = dict(map(lambda item: (item[0], mapping[item[1]]), y_true.items()))
y_pred = dict(map(lambda item: (item[0], item[1][0]), y_pred.items()))



#print(choices,y_pred)
acc = score(y_true,y_pred)
print(acc)

save_as_json(y_true,'y_true')
save_as_json(y_pred,'y_pred')