from keras.models import Model
import tensorflow as tf
import app.models as m


def test_glove_model():

    test_model = m.glove_model(vector_dim=15, vocab_size=10)

    assert type(test_model) == Model
    assert len(test_model.layers) == 11

def test_0_in_loss():
    losses = m.custom_loss(
        y_true=tf.constant([0.1, 0, 0]), 
        y_pred=tf.constant([0, 0, 0.1])
    )

    assert not tf.reduce_any(tf.math.is_nan(losses))
