from utils import gpt3,add_response,score
import os

# Expose the API_KEY to the system
os.environ['OPENAI_API_KEY'] = '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'

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

#print(choices,y_pred)
acc = score(choices,y_pred)
print(acc)    