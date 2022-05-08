# DS203_Project

This is the course final project for DS-UA 203 Machine Learning for Language Understanding, taught by Professor [Sam Bowman](https://cims.nyu.edu/~sbowman/) at Spring 2022.

## Team Members
[Yukai Yang](https://www.linkedin.com/in/yukai-yang-7bba651a3/)

[Tracy Zhu](https://www.linkedin.com/in/yixin-zhu-701478188/)

[Yuchen Zhu](https://yuchen-zhu-zyc.github.io/)

To download the whole directory, enter

``git clone https://github.com/yk803/DS203_Project.git``

in terminal.

## Code Instructions
To run the code, first run

``bash env.sh``

to get the necessary packages and download the raw dataset (if there isn't one).

For selecting words and sentences, follow the insturctions in `generate_dataset.ipynb` to build the dataset.

Next, call 

`python add_fake_words.py data_dir mode` 

to generate the task dataset. 

`data_dir`: the path name of the dataset we just built.

`mode`: how we want our fake new words to be generated.

Finally, call 

`python main.py API_key mode`

to send requests to access OpenAI GPT-3, with your unique API key. Note the key may rotate and may thus need updates.

`API_key`: the API key to access GPT-3

`mode`: how we generated the fake new words.
