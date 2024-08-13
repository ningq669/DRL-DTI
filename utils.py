import os
import random
import numpy as np
import torch
import dgl
import logging
import scipy.io as sio

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def load_network_data(name):
    net = sio.loadmat('./data/' + name + '.mat')
    X, A, Y = net['attrb'], net['network'], net['group']
    if name in ['cs', 'photo']:
        Y = Y.flatten()
        Y = one_hot_encode(Y, Y.max() + 1).astype(np.int)
    return A, X, Y


def random_planetoid_splits(num_classes, y, train_num, seed):
    # Set new random planetoid splits:
    # *  train_num * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    np.random.seed(seed)
    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_num] for i in indices], dim=0)

    rest_index = torch.cat([i[train_num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    val_index = rest_index[:500]
    test_index = rest_index[500:1500]

    return train_index, val_index, test_index


def get_train_data(labels, tr_num, val_num, seed):
    np.random.seed(seed)
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max() + 1

    idx_train = []
    idx_val = []
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        pos0 = np.random.permutation(pos0)
        idx_train.append(pos0[0:tr_num])
        idx_val.append(pos0[tr_num:val_num + tr_num])

    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = np.setdiff1d(range(labels.shape[0]), np.union1d(idx_train, idx_val))

    idx_train = torch.LongTensor(np.random.permutation(idx_train))
    idx_val = torch.LongTensor(np.random.permutation(idx_val))
    idx_test = torch.LongTensor(np.random.permutation(idx_test))

    return idx_train, idx_val, idx_test

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func1(x):
    # d = zip(*x)
    d = dgl.batch(x)
    return d

def graph_collate_func2(x):
    # p = zip(*x)
    return torch.tensor(np.array(x))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding



