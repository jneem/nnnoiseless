import numpy as np
import re

def _append_vector(bs, vector):
    vector = np.reshape(vector, (-1))
    # Weights should be in the range -0.5, 0.5; we convert that to a signed byte. It seems like the
    # python bytearray only likes unsigned bytes, so we need to compute the unsigned twos-complement representation
    # of the signed byte we actually want.
    def b(x):
        y = np.clip(int(round(256 * x)), -128, 127)
        if y < 0:
            y = 256 + y
        return y

    v = [b(x) for x in vector]
    bs.extend(v)

def _activation(layer):
    name = re.search('function (.*) at', str(layer.activation)).group(1).upper()
    if name == 'SIGMOID':
        return 1
    elif name == 'RELU':
        return 2
    else:
        return 0

def _append_layer(bs, layer):
    weights = layer.get_weights()
    act = _activation(layer)
    nb_inputs = weights[0].shape[0]
    nb_neurons = weights[0].shape[1]

    if len(weights) > 2:
        # This is a GRU layer.
        nb_neurons = int(nb_neurons / 3)

    bs.extend([nb_inputs, nb_neurons, act])
    for i in range(0, len(weights)):
        _append_vector(bs, weights[i])


def dump_model(model, filename):
    with open(filename, 'wb') as file:
        bs = bytearray()
        for i, layer in enumerate(model.layers):
            if len(layer.get_weights()) > 0:
                _append_layer(bs, layer)
        file.write(bs)

