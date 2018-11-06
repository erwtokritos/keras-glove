This repo contains an implementation of the popular [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) in Python with
Keras/Tensorflow.


Introduction
------
Similarly to [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 
, GloVe is an unsupervised algorithm which learns vector representations for words. It is trained on aggregated 
word-word co-occurrence statistics and the resulting vectors expose linear substructures. For more detailed information
the interested user should refer to the [original web site](https://nlp.stanford.edu/projects/glove/)


#### Implemented so far
1. Model architecture in Keras with a custom loss function
2. Helper functions for data loading and transformations 


#### Pending
- Unit tests
- A new version with less memory requirements. 



Setup
----

### Datasets
You need to download a corpus in order to run the algorithm. Some suggestions:
- http://mattmahoney.net/dc/text8.zip
- [Project Gutenberg](https://www.gutenberg.org/ebooks/)
- https://github.com/kavgan/nlp-text-mining-working-examples/blob/master/word2vec/reviews_data.txt.gz

The files should be unzipped and by convention are placed into the `data` folder


### Output
All the generated files are stored in the `output` folder 


#### Requirements
- Python 3.6
- virtualenv

Start a new virtual environment as:
```bash
virtualenv --no-site-packages -p python3.6 venv
```

and then activate it with
```bash
. venv/bin/activate
```

You can install the library either with
```bash
python setup.py install
```

or 
```bash
python setup.py develop
```
if you want to stay up to date


Running the code
---
Currently there are two commands available. The first performs the actual training based on
the given corpus. For example, the command:
```bash
kglove train data/my_corpus.txt -n 5000 -v 30 -e 3 -b 4098
```
will read the first 5000 lines from `my_corpus.txt`, it will train the model for 3 epochs with batch size 4098 producing 
30-dimensional vectors. 
Look in `orchestrator.py` for the different options available


The second command returns the closest neighbours for a (comma separated) list of words:
```bash
kglove closest dirty,polite
```
will produce the following output:
```bash
Most similar words to dirty:
[('laid', 0.9819546), ('smelled', 0.98293394), ('dark', 0.98691344), ('worn', 0.99096316)]:

Most similar words to polite:
[('accomodating', 0.9964925), ('courteous', 0.9968112), ('professional', 0.9969903), ('accommodating', 0.99781287)]:

```