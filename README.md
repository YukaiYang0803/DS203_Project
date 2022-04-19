# DS203_Project

This is the course final project for DS-UA 203 Machine Learning for Language Understanding, taught by Professor [Sam Bowman](https://cims.nyu.edu/~sbowman/) at Spring 2022.

## Team Members
[Yukai Yang](https://github.com/yk803)
Tracy Zhu
Yuchen Zhu

To download the whole directory, enter

``git clone https://github.com/yk803/DS203_Project.git``

in terminal.

## Code Instructions
To run the code, first run

``bash env.sh``

to get the necessary packages and download the raw dataset (if there isn't one).

For selecting words and sentences, follow the insturctions in `generate_dataset.ipynb` to build the dataset.

Next, call `python add_fake_words.py` to generate the task dataset. It has two arguments: `data_dir`, representing the path name of the dataset we just built, and `mode`, determining how we want our fake new words to be generated.

Finally, call `python main.py` to send requests to access OpenAI GPT-3, with your unique API key. The key may rotate sometimes and may thus need updates.
