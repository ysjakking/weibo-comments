from numpy.random import RandomState
from random import Random
import theano

from theano.tensor.shared_randomstreams import RandomStreams as RandomStreams_cpu

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


seed = 6666

py_rng = Random(seed)
np_rng = RandomState(seed)
t_rng = RandomStreams(seed)
t_rng_cpu=RandomStreams_cpu(seed)

def set_seed(n):
    global seed, py_rng, np_rng, t_rng
    
    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)
    t_rng = RandomStreams(seed)
