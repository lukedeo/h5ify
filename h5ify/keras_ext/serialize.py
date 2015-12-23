#!/usr/bin/env python
'''
serialize.py -- just a simple way to save both the model config
and the weights to the same place. Useful for architecture experimentation

author: Luke de Oliveira (lukedeo@stanford.edu)
'''

import logging

from keras.models import model_from_json

from ..generic import save as _save
from ..generic import load as _load

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def save_model(net, filename):
    '''
    save_model -- saves a keras model *including the config* to an 
    hdf5 file.

    Args:
    -----
        net: a Keras model to save.
        filename: file to create to hold the serialized model.

    Raises:
    -------
        raises a TypeError if the object to be saved doesn't have
        the `to_json` and `get_weights` attr/functions.
    '''

    if (not hasattr(net, 'to_json')) or (not hasattr(net, 'get_weights')):
        raise TypeError(
            'passed parameter `net` appears not to be a keras model!')

    payload = {
        'config' : net.to_json(),
        'weights' : net.get_weights()
    }

    _save(filename, payload, compress=True)

def load_model(filename, *args, **kwargs):
    '''
    load_model -- loads a serialized keras model

    Args:
    -----
        filename: file to load net from
        *args, **kwargs: additional arguments passed to model.compile(). 
            If these are empty, no compilation is done.
    Returns:
    --------
        the network instantiated with weights. Net is compiled iff valid 
        *args and **kwargs are passed for the network compilation process
    '''

    data = _load(filename)
    net = model_from_json(data['config'])
    net.set_weights(data['weights'])

    if (len(kwargs) + len(kwargs)) == 0:
        return net

    try:
        _logger.info('compiling model from file: {}'.format(filename))
        net.compile(*args, **kwargs)
    except Exception:
        _logger.error(
            'error in model compilation. Returning uncompiled model.')
    return net



