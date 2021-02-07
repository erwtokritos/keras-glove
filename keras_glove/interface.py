from collections import defaultdict

import click

from keras_glove.models import glove_model
from keras_glove.save_utils import save_model
from keras_glove.evaluators import get_most_similar
from keras_glove import text_utils as tu
from datasets import load_dataset


@click.group()
def cli():
    pass


@cli.command()
@click.argument('filename', type=click.File('rb'))
@click.option('-n', '--num-lines', help='Number of lines to read from filename. -1 means that all lines are used',
              default=-1)
@click.option('-v', '--vector-size', help='Dimension of the generated embeddings', default=30)
@click.option('-w', '--window', help='The size of the window around the central word', default=5)
@click.option('-e', '--epochs', help='Number of epochs to execute', default=5)
@click.option('-b', '--batch-size', help='The size of the batch', default=512)
@click.option('--num-words', help='The number of (most frequent) words to keep', default=10000)
def train(filename, num_lines, vector_size, window, epochs, batch_size, num_words):
    click.echo(click.style('Loading data', fg='green'))

    # read file and tokenize
    lines = tu.read_file(filename, num_lines=num_lines)
    seqs, tokenizer = tu.tokenize(lines=lines, num_words=num_words)

    # initialize cache
    cache = defaultdict(lambda: defaultdict(int))

    # get co-occurrences
    click.echo(click.style('Computing co-occurrences', fg='green'))
    tu.build_cooccurrences(sequences=seqs, cache=cache, window=window)
    first_indices, second_indices, frequencies = tu.cache_to_pairs(cache=cache)

    # build GloVe model & fit
    click.echo(click.style('Training keras model', fg='green'))
    model = glove_model(tokenizer.num_words + 1, vector_dim=vector_size)
    model.fit([first_indices, second_indices], frequencies, epochs=epochs, batch_size=batch_size)

    # save embedding layers as numpy arrays
    click.echo(click.style('Saving data', fg='green'))
    save_model(model=model, tokenizer=tokenizer)

    click.echo(click.style('Complete', fg='green'))


@cli.command()
@click.option('-n', '--num-lines', help='Number of lines to read from filename. -1 means that all lines are used',
              default=50000)
@click.option('-v', '--vector-size', help='Dimension of the generated embeddings', default=30)
@click.option('-w', '--window', help='The size of the window around the central word', default=5)
@click.option('-e', '--epochs', help='Number of epochs to execute', default=4)
@click.option('-b', '--batch-size', help='The size of the batch', default=4096)
@click.option('--num-words', help='The number of (most frequent) words to keep', default=10000)
@click.option('--hf_ds', help='The dataset to use provided by the Huggingface team (https://huggingface.co/datasets)', default='cnn_dailymail')
@click.option('--hf_ds_version', help='The version of the dataset', default=None)
@click.option('--hf_split', help='The split of the dataset', default='train')
@click.option('--hf_label', help='The label containing the text portion of the item', default=None)
def hf_train(num_lines, vector_size, window, epochs, batch_size, num_words, hf_ds, hf_ds_version, hf_split, hf_label):
    click.echo(click.style('* Loading dataset..', fg='green'))

    if hf_ds_version:
        _ds = load_dataset(hf_ds, hf_ds_version)
    else:
        _ds = load_dataset(hf_ds)

    lines = _ds[hf_split][:num_lines][hf_label]

    # read file and tokenize
    seqs, tokenizer = tu.tokenize(lines=lines, num_words=num_words)

    # initialize cache
    cache = defaultdict(lambda: defaultdict(int))

    # get co-occurrences
    click.echo(click.style('* Computing co-occurrences', fg='green'))
    tu.build_cooccurrences(sequences=seqs, cache=cache, window=window)
    first_indices, second_indices, frequencies = tu.cache_to_pairs(cache=cache)

    # build GloVe model & fit
    click.echo(click.style('* Training keras model', fg='green'))
    model = glove_model(tokenizer.num_words + 1, vector_dim=vector_size)
    model.fit([first_indices, second_indices], frequencies, epochs=epochs, batch_size=batch_size)

    # save embedding layers as numpy arrays
    click.echo(click.style('* Saving data', fg='green'))
    save_model(model=model, tokenizer=tokenizer)

    click.echo(click.style('Complete', fg='green'))


@cli.command()
@click.argument('words', type=click.STRING)
@click.option('-k', type=int, help='The number of most similar to retrieve', default=5)
def closest(words, k):

    for word in words.split(','):
        res = get_most_similar(word=word, k=k)
        if res:
            click.echo(click.style(f'Most similar words to {word}:', fg='green'))
            click.echo(click.style(f'{res}\n', fg='green'))
        else:
            click.echo(click.style(f'Word {word} not in the vocabulary', fg='red'))
