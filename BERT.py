import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer
import random
from sklearn.metrics import accuracy_score as ac
import argparse

parser = argparse.ArgumentParser(description='Generate the query and run the experiment.')

parser.add_argument('mode',
                    default=1,
                    type=int,
                    help='Choose the new words generate mode')

parser.add_argument('few_shot',
                    default=0,
                    type=int,
                    help='Choose whether to use few-shot')

parser.add_argument("model",
                    default='roberta',
                    type=str,
                    help="Choose class of model")

args = parser.parse_args()

mode = args.mode
few_shot = args.few_shot
model_class = args.model
if mode == 1:
    mode = '_typo'
elif mode == 2:
    mode = '_prefix_suffix'
elif mode == 3:
    mode = '_replace_pre_suffix'
elif mode == 4:
    mode = '_random'


def generate_query(data,idx,mode="_replace_pre_suffix"):
    choices = [data['word'],data['choice1'],data['choice2'],data['choice3']]
    # When the dataset is under construction: only generate query for those that are finished
    if pd.isnull(choices[-1]):
        return None,None
    # shuffle the choices (so the answer is not always A)
    random.seed(idx)
    random.shuffle(choices)
    query_sens = "Read the following sentences\n" + data['sen1'+mode] + "\n" + data['sen2'+mode] + "\n" + data['sen3'+mode] + "\n"
    #query_sens = data['sen1'+mode] + "\n" # one example sentence
    query_ques = f"Which of the following explains the word {data['mask'+mode]} in the sentence better\n"
    query_opts = "Provide the answer in A, B, C, or D.\n"
    return query_sens+query_ques+query_opts, choices.index(data['word']), choices


shot = """
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


PATH = "model_cp/{}/".format(model_class)
tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model = AutoModelForMultipleChoice.from_pretrained(PATH, local_files_only=True)
file = "data{}.csv".format(mode)
df = pd.read_csv(file)
ground_truth = []
prediction = []
for idx in range(len(df)):
    row = df.iloc[idx]
    prompt, y_true, choices = generate_query(row, idx, mode)
    ground_truth.append(y_true)
    if few_shot:
        prompt = shot + prompt
    encoding = tokenizer([prompt,prompt,prompt, prompt], choices, return_tensors="pt", padding=True)
    labels = torch.tensor(y_true).unsqueeze(0)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)
    logits = outputs.logits
    y_pred = torch.argmax(logits).item()
    prediction.append(y_pred)

score = ac(ground_truth, prediction)
with open("result.txt", "a") as f:
    f.write("{}, {}, {}, {}\n".format(model_class, mode, few_shot, score))
print(model_class, mode, few_shot, score)
