from __future__ import division, print_function, absolute_import

try:
    import tables
    _pytables_ok = True
    del tables
except ImportError:
    _pytables_ok = False

if _pytables_ok:
    from .h5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function")
    load = save = _f

__all__ = ['load', 'save']
