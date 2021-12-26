#!/usr/bin/python

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import load_model
from keras import backend as K
import sys
import re
import numpy as np

def append_vector(bs, vector):
    vector = np.reshape(vector, (-1))
    # Weights should be in the range -0.5, 0.5; we convert that to a byte in range(0, 256)
    # FIXME: two complement?
    v = [np.clip(int(round(256 * (x + 0.5))), 0, 255) for x in vector]
    print(f"len: {len(v)}");
    bs.extend(v)

def activation(layer):
    name = re.search('function (.*) at', str(layer.activation)).group(1).upper()
    if name == 'SIGMOID':
        return 1
    elif activation == 'RELU':
        return 2
    else:
        return 0

def append_layer(bs, layer):
    weights = layer.get_weights()
    act = activation(layer)
    nb_inputs = weights[0].shape[0]
    nb_neurons = weights[0].shape[1]

    if len(weights) > 2:
        # This is a GRU layer.
        nb_neurons = int(nb_neurons / 3)

    print([nb_inputs, nb_neurons, act])
    bs.extend([nb_inputs, nb_neurons, act])
    for i in range(0, len(weights)):
        append_vector(bs, weights[i])

def printVector(f, ft, vector, name):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const rnn_weight {}[{}] = {{\n   '.format(name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(min(127, int(round(256*v[i])))))
        ft.write('{}'.format(min(127, int(round(256*v[i])))))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break;
        ft.write(" ")
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    ft.write("\n")
    return;

def printLayer(f, ft, layer):
    weights = layer.get_weights()
    activation = re.search('function (.*) at', str(layer.activation)).group(1).upper()
    if len(weights) > 2:
        ft.write('{} {} '.format(weights[0].shape[0], int(weights[0].shape[1]/3)))
    else:
        ft.write('{} {} '.format(weights[0].shape[0], weights[0].shape[1]))
    if activation == 'SIGMOID':
        ft.write('1\n')
    elif activation == 'RELU':
        ft.write('2\n')
    else:
        ft.write('0\n')
    printVector(f, ft, weights[0], layer.name + '_weights')
    if len(weights) > 2:
        printVector(f, ft, weights[1], layer.name + '_recurrent_weights')
    printVector(f, ft, weights[-1], layer.name + '_bias')
    name = layer.name
    if len(weights) > 2:
        f.write('static const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
                .format(name, name, name, name, weights[0].shape[0], weights[0].shape[1]/3, activation))
    else:
        f.write('static const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
                .format(name, name, name, weights[0].shape[0], weights[0].shape[1], activation))

def structLayer(f, layer):
    weights = layer.get_weights()
    name = layer.name
    if len(weights) > 2:
        f.write('    {},\n'.format(weights[0].shape[1]/3))
    else:
        f.write('    {},\n'.format(weights[0].shape[1]))
    f.write('    &{},\n'.format(name))


def foo(c, name):
    return None

def mean_squared_sqrt_error(y_true, y_pred):
    return K.mean(K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


model = load_model(sys.argv[1], custom_objects={'msse': mean_squared_sqrt_error, 'mean_squared_sqrt_error': mean_squared_sqrt_error, 'my_crossentropy': mean_squared_sqrt_error, 'mycost': mean_squared_sqrt_error, 'WeightClip': foo})

weights = model.get_weights()

f = open(sys.argv[2], 'w')
ft = open(sys.argv[3], 'w')

f.write('/*This file is automatically generated from a Keras model*/\n\n')
f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "rnn.h"\n#include "rnn_data.h"\n\n')
ft.write('rnnoise-nu model file version 1\n')

bs = bytearray()
layer_list = []
for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:
        printLayer(f, ft, layer)
        append_layer(bs, layer)
    if len(layer.get_weights()) > 2:
        layer_list.append(layer.name)

test_f = open('out.rnn', 'wb')
test_f.write(bs)

f.write('const struct RNNModel rnnoise_model_{} = {{\n'.format(sys.argv[4]))
for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:
        structLayer(f, layer)
f.write('};\n')

#hf.write('struct RNNState {\n')
#for i, name in enumerate(layer_list):
#    hf.write('  float {}_state[{}_SIZE];\n'.format(name, name.upper())) 
#hf.write('};\n')

f.close()
