from deepctr.layers.activation import Dice
from deepctr.tests.utils import layer_test


def test_dice():
    layer_test(Dice, kwargs={'num_features': 3, 'dim': 2}, input_shape=(5, 3), expected_output_shape=(5, 3))
    layer_test(Dice, kwargs={'num_features': 10, 'dim': 3}, input_shape=(5, 3, 10), expected_output_shape=(5, 3, 10))
    layer_test(Dice, kwargs={'num_features': 3, 'dim': 2},
               input_shape=(5, 3), expected_output_shape=(5, 3), fixed_batch_size=True)