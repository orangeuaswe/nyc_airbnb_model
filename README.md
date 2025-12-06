## install dependencies 

run:
pip install -r reqs.txt
python -m nltk.downloader vader_lexicon

## how to train model 

1. go into the data folder and run python merge.py to merge all the datasets into one cohesive file.
2. go back and then go into backend, and then run python train.py.
3. to visualize this, run python visual.py, and it'll generate a heatmap of the data.


