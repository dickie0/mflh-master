import sys
import numpy as np
from utils.util import sign


def euclidean2(x1, x2):
    return np.sum(np.square(x1 - x2), axis=-1)


def inner_product(x1, x2, pair=False):
    if pair:
        return - np.inner(x1, x2)
    else:
        return - np.sum(x1 * x2, axis=-1)


def metric(x1, x2=None, pair=True, dist="euclidean2", sgn=False):
    if x2 is None:
        x2 = x1
    if sgn:
        x1 = sign(x1)
        x2 = sign(x2)
    if dist == 'inner_product':
        return inner_product(x1, x2, pair)
    if pair:
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 0)
    return getattr(sys.modules[__name__], dist)(x1, x2)
