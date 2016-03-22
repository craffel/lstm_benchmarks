import scipy.io
import theano
import theano.tensor as T
import numpy as np

import logging

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('experiment.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

TRAIN_NC = '../data/train_1_speaker.nc'
VAL_NC = '../data/val_1_speaker.nc'
BATCH_SIZE = 50


def one_hot(labels, n_classes):
    '''
    Converts an array of label integers to a one-hot matrix encoding

    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes

    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    '''
    one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
    one_hot[range(labels.shape[0]), labels] = True
    return one_hot


def load_netcdf(filename):
    '''
    Loads in data from a netcdf file in rnnlib format

    :parameters:
        - filename : str
            Path to a netcdf file

    :returns:
        - X : list of np.ndarray
            List of time series matrices
        - y : list of np.ndarray
            List of label arrays in one-hot form (see one_hot)
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        netcdf_data = scipy.io.netcdf_file(f).variables

    X = []
    y = []
    n = 0
    for length in netcdf_data['seqLengths'].data:
        X_n = netcdf_data['inputs'].data[n:n + length]
        X.append(X_n.astype(theano.config.floatX))
        y_n = one_hot(netcdf_data['targetClasses'].data[n:n + length],
                      netcdf_data['numTargetClasses'].data)
        y.append(y_n.astype(theano.config.floatX))
        n += length
    return X, y


def make_batches(X, length, batch_size=BATCH_SIZE):
    '''
    Convert a list of matrices into batches of uniform length

    :parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            matrix
    '''
    n_batches = len(X)//batch_size
    X_batch = np.zeros((n_batches, batch_size, length, X[0].shape[1]),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_batches, batch_size, length), dtype=np.bool)
    for b in range(n_batches):
        for n in range(batch_size):
            X_m = X[b*batch_size + n]
            X_batch[b, n, :X_m.shape[0]] = X_m[:length]
            X_mask[b, n, :X_m.shape[0]] = 1
    return X_batch, X_mask


logger.info('Loading data...')
X_train, y_train = load_netcdf(TRAIN_NC)
X_train = X_train
y_train = y_train
X_val, y_val = load_netcdf(VAL_NC)
X_val = X_val
y_val = y_val

# Find the longest sequence
length = max(max([X.shape[0] for X in X_train]),
             max([X.shape[0] for X in X_val]))
# Convert to batches of time series of uniform length
X_train, _ = make_batches(X_train, length)
y_train, train_mask = make_batches(y_train, length)
X_val, _ = make_batches(X_val, length)
y_val, val_mask = make_batches(y_val, length)

# Create a Keras model and train and evaluate it on the data.
from keras.models import Graph
from keras.layers import LSTM, GaussianNoise, TimeDistributedDense, Activation
from keras.optimizers import SGD
model = Graph()
model.add_input(name='input', input_shape=(length,))
# TODO Mask input?
model.add_node(GaussianNoise(0.6), name='noise', input='input')
model.add_node(LSTM(input_dim=78, output_dim=150, return_sequences=True), name='forward_1', input='noise')
model.add_node(LSTM(input_dim=78, output_dim=150, return_sequences=True, go_backwards=True), name='backward_1', input='noise')
model.add_node(LSTM(input_dim=150, output_dim=51, return_sequences=True), name='forward_2', inputs=['forward_1','backward_1'], merge_mode='sum')
model.add_node(LSTM(input_dim=150, output_dim=51, return_sequences=True, go_backwards=True), name='backward_2', inputs=['forward_1','backward_1'], merge_mode='sum')
model.add_node(LSTM(input_dim=51, output_dim=51, return_sequences=True), name='forward_3', inputs=['forward_2','backward_2'], merge_mode='sum')
model.add_node(LSTM(input_dim=51, output_dim=51, return_sequences=True, go_backwards=True), name='backward_3', inputs=['forward_2','backward_2'], merge_mode='sum')
model.add_node(TimeDistributedDense(input_dim=51), name='time_distributed_dense', inputs=['forward_3', 'backward_3'], merge_mode='sum')
model.add_node(Activation('softmax'), name='activation', input='time_distributed_dense')
model.add_output(name='output', input='activation')
model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.02, momentum=0.9))
model.fit({'input': X_train, 'output': y_train}, batch_size=BATCH_SIZE, nb_epoch=500, show_accuracy=True, validation_data=(X_val, y_val))
