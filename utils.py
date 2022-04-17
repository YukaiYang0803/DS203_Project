
import os
import openai


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
    response = gpt3(query)
    ans = response.strip()
    return ans

def score(y_true,y_pred):
    total = len(y_true)
    return 100*sum([y_true[key]==y_pred[key] for key in y_true.keys()])/total
