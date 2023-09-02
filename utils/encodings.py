import numpy as np
from cell_module.ops import OPS as ops_dict

INPUT = 'input'
OUTPUT = 'output'
OPS = list(ops_dict.keys())
OPS.remove('bottleneck')

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def encode_paths(path_indices):
    """ output one-hot encoding of paths """
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding