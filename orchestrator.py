from collections import defaultdict

import click

from app.models import glove_model
from app.save_utils import save_model
from app.evaluators import get_most_similar
from app import text_utils as tu


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
@click.argument('words', type=click.STRING)
@click.option('-k', type=int, help='The number of most similar to retrieve', default=5)
def closest(words, k):

    for word in words.split(','):
        res = get_most_similar(word=word, k=k)
        if res:
            click.echo(click.style(f'Most similar words to {word}:', fg='green'))
            click.echo(click.style(f'{res}:\n', fg='green'))
        else:
            click.echo(click.style(f'Word {word} not in the vocabulary', fg='red'))
