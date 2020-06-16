from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from app.config import *


X_MAX = 100
a = 3.0 / 4.0


def glove_model(vocab_size=10, vector_dim=3):
    """
    A Keras implementation of the GloVe architecture
    :param vocab_size: The number of distinct words
    :param vector_dim: The vector dimension of each word
    :return:
    """
    input_target = Input((1,), name='central_word_id')
    input_context = Input((1,), name='context_word_id')

    central_embedding = Embedding(vocab_size, vector_dim, input_length=1, name=CENTRAL_EMBEDDINGS)
    central_bias = Embedding(vocab_size, 1, input_length=1, name=CENTRAL_BIASES)

    context_embedding = Embedding(vocab_size, vector_dim, input_length=1, name=CONTEXT_EMBEDDINGS)
    context_bias = Embedding(vocab_size, 1, input_length=1, name=CONTEXT_BIASES)

    vector_target = central_embedding(input_target)
    vector_context = context_embedding(input_context)

    bias_target = central_bias(input_target)
    bias_context = context_bias(input_context)

    dot_product = Dot(axes=-1)([vector_target, vector_context])
    dot_product = Reshape((1, ))(dot_product)
    bias_target = Reshape((1,))(bias_target)
    bias_context = Reshape((1,))(bias_context)

    prediction = Add()([dot_product, bias_target, bias_context])

    model = Model(inputs=[input_target, input_context], outputs=prediction)
    model.compile(loss=custom_loss, optimizer=Adam())

    return model


def custom_loss(y_true, y_pred):
    """
    This is GloVe's loss function
    :param y_true: The actual values, in our case the 'observed' X_ij co-occurrence values
    :param y_pred: The predicted (log-)co-occurrences from the model
    :return: The loss associated with this batch
    """
    return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) * K.square(y_pred - K.log(1 + y_true)), axis=-1)
