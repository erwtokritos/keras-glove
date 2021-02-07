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


Setup
----

#### Requirements
- Python 3.8
- poetry (https://python-poetry.org/)


Create a virtual environment with `poetry`:
```bash
poetry install
```

and then activate it with
```bash
poetry shell
```


Running the code
---
There are three commands. 


The first command trains the model on a corpus provided by the user, e.g. a Kaggle dataset, where there is a text item (e.g. tweet, article etc) per line. We train GloVe, with:
```bash
 kglove train data/<our corpus>.csv -v 20 -e 5 -b 64
```
which will train the model for 5 epochs (`-e` option) with batch size 64 (`-b` option) producing 
20-dimensional vectors (`-v` option). 
Other options are:
- `-n` or `--num-lines`: Number of lines to read from a file. Useful when you want to run a test on a smaller version of the dataset 
- `-w` or `--window`: The window parameter required by GloVe (by default it is 5)
- `--num-words` : The number of most frequent words that will form the vocabulary

Take a look at `keras_glove/interface.py` for more info 

---

The second training command takes advantage of the [dataset collection](https://github.com/huggingface/datasets) provided by 
the awesome [Huggingface team](https://huggingface.co/). There are additional options (compared to the above)
which specify the dataset configuration. More specifically:

- `--hf_ds`: The dataset to use (e.g. `cnn_dailymail`)
- `--hf_version`: The version of the dataset (not mandatory)
- `--hf_split`: The split (e.g. `train`, `validation` etc).
- `--hf_label`: The label of the text item. Dataset-specific
```bash
 kglove hf-train --hf_ds cnn_dailymail  --hf_split train  --hf_ds_version 3.0.0 --hf_label article -v 30 -e 4
```

Both commands will save the trained model to the `output` folder.

* Note: Each run overwrites previous runs 


---
The third command loads the trained model and returns the closest neighbours for a (comma separated) list of input words:
For example, training on a small portion of the `cnn_dailymail` dataset as presented above we get:

```bash
>  kglove closest russia,2008

Most similar words to russia:
[('korea', 0.9835626), ('syria', 0.98309094), ('afghanistan', 0.9797111), ('iraq', 0.9768379), ('ukraine', 0.9753001)]

Most similar words to 2008:
[('2006', 0.9922105), ('2012', 0.9903844), ('2010', 0.9899334), ('2009', 0.98728645), ('2007', 0.9869324)]

```

```bash
>  kglove closest clinton

Most similar words to clinton:
[('romney', 0.98541075), ('bush', 0.97941786), ('president', 0.9757596), ('barack', 0.96643496), ('press', 0.9561945)]
```

There is an extra option `-k` to specify the number of returned items (by default it is 5)