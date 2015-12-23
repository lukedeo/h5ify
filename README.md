# `h5ify`

This repo is essentially a stripped down version of [`deepdish`](https://github.com/uchicago-cs/deepdish) with some simple additions to make keeping track of [`keras`](https://github.com/fchollet/keras) models a bit easier. The entire code in `generic/` is from [`deepdish`](https://github.com/uchicago-cs/deepdish), I just took out the relevant stuff.

To save/load keras models, you can use:

```python
from h5ify.keras_ext import save_model, load_model
# assume you have a Keras model
net = MyFancyModel()

save_model(net, './mynet.h5')

# loads uncompiled
nn = load_model('./mynet.h5')

# can load a compiled instance!
nn_other = load_model('./mynet.h5', 'sgd', 'mse', class_mode='categorical')
```

To install it, just clone and do the usual `python setup.py build` and `python setup.py install`, with sudo's as needed. Or, just do `pip install git+https://github.com/lukedeo/h5ify.git`.

