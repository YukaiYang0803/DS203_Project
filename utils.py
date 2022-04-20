import os
import openai
import random
import pandas as pd
import json

def gpt3(gtext,print_content=False):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=gtext,
      temperature=0,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
    )
    content = response.choices[0].text.split(' ');
    if print_content:
        print(content)
    return response.choices[0].text

# Helper Functions

def add_response(query):
    """ Call the GPT-3 model with query, and record the response choice. """
    response = gpt3(query)
    ans = response.strip()
    return ans

def score(y_true,y_pred):
    """ Compute acc of the prediction"""
    total = len(y_true)
    return 100*sum([y_true[key]==y_pred[key] for key in y_true.keys()])/total

# Generate task mode 1: Add typo
def make_typo(word,idx):
    # fix a random state so that the random change is reproducible
    random.seed(idx)
    length = len(word)
    # if the word is too short (length == 3) (no single or two-letter word in the dataset)
    # we make the typo easier: just duplicate the first and the last letter
    if length == 3:
        return word[0]*2+word[1]+word[2]*2
    # choose letters at which indices to be changed
    # set the number of changes (typos) to 3
    # And sort the indices so the typo maker below will work sequentially correct
    idx_to_change = sorted(random.sample(range(1,length),3))
    # Decide how to change (True ==> duplicate | False ==> remove)
    # fix random state to maintain the same result
    random.seed(idx)
    how_to_change = [bool(random.getrandbits(1)),bool(random.getrandbits(1)),bool(random.getrandbits(1))]
    new_word = ''
    prev = 0
    for idx, boo in zip(idx_to_change, how_to_change):
        curr = idx
        new_word += word[prev:curr]
        if boo: # duplicate
            new_word += word[idx]*2
        else: # delete
            pass
        # Next iteration start from the next char
        prev = idx + 1
    if idx < length - 1: # not the last index: adding back the last part of the original word
        new_word += word[prev:]
    if len(new_word) < 3: # after making typo the word becomes too short
        new_word = word[0]*2+word[1]+word[2:]*2
    return new_word


def make_pre_suffix(word,idx,include_embedded=False,prefix_file='prefix.json',embedded_file='embedded.json',suffix_file='suffix.json'):
    def load_morpheme(file):
        with open(file) as f:
            data = json.load(f)
            f.close()
            return data
    prefix = load_morpheme(prefix_file)
    embedded = load_morpheme(embedded_file)
    suffix = load_morpheme(suffix_file)
    # fix a random state so that the random change is reproducible
    random.seed(idx)
    idx_p,idx_e,idx_s = random.randint(0,len(prefix)-1),random.randint(0,len(embedded)-1),random.randint(0,len(suffix)-1)
    if include_embedded:
        new_word = prefix[str(idx_p)]+ embedded[str(idx_e)] + suffix[str(idx_s)]
    else:
        new_word = prefix[str(idx_p)]+ suffix[str(idx_s)]
    return new_word

# function to extract prefix, embedded, and suffix morphemes
def get_pre_suffix_data(morpheme_file):
    prefix,embedded,suffix = {},{},{}
    idx_p,idx_e,idx_s = 0,0,0
    with open(morpheme_file) as f:
        nn = json.load(f)
    for word in nn.keys():
        for form in nn[word]['forms']:
            if form['loc'] == 'prefix':
                prefix[idx_p] = form['form']
                idx_p += 1
            elif form['loc'] == 'embedded':
                embedded[idx_e] = form['form']
                idx_e += 1
            elif form['loc'] == 'suffix':
                suffix[idx_s] = form['form']
                idx_s += 1
    save_as_json(prefix,'prefix')
    save_as_json(embedded,'embedded')
    save_as_json(suffix,'suffix')


def change_word(word,idx,mode):
    if mode == 1:
        return make_typo(word,idx)
    elif mode == 2:
        return make_pre_suffix(word,idx)

def generate_query(data,idx,mode='_typo'):
    choices = [data['word'],data['choice1'],data['choice2'],data['choice3']]
    # When the dataset is under construction: only generate query for those that are finished
    if pd.isnull(choices[-1]):
        return None,None
    # shuffle the choices (so the answer is not always A)
    random.seed(idx)
    random.shuffle(choices)
    #query_sens = "Read the following sentences\n" + data['sen1'+mode] + "\n" + data['sen2'+mode] + "\n" + data['sen3'+mode] + "\n"
    query_sens = data['sen1'+mode] + "\n" # one example sentence
    query_ques = f"Which of the following explains the word {data['mask'+mode]} in the sentence better\n"
    query_opts = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nProvide the answer in A, B, C, or D.\n "
    return query_sens+query_ques+query_opts, choices.index(data['word'])

def save_as_json(dictionary,name):
    with open(f"{name}.json", "w") as outfile:
        json.dump(dictionary, outfile, indent=4)
        outfile.close()