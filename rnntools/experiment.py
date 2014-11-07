import scipy.io
import nntools
import rnntools
import theano
import theano.tensor as T
import numpy as np
import time

TRAIN_NC = '../data/train_1_speaker.nc'
VAL_NC = '../data/val_1_speaker.nc'


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
    with open(filename, 'r') as f:
        netcdf_data = scipy.io.netcdf_file(f).variables

    X = []
    y = []
    n = 0
    for length in netcdf_data['seqLengths'].data:
        X.append(netcdf_data['inputs'].data[n:n + length])
        y.append(one_hot(netcdf_data['targetClasses'].data[n:n + length],
                         netcdf_data['numTargetClasses'].data))
        n += length
    return X, y

print 'Loading data...'
X_train, y_train = load_netcdf(TRAIN_NC)
X_train = X_train
y_train = y_train
X_val, y_val = load_netcdf(VAL_NC)
X_val = X_val
y_val = y_val

n_epochs = 50
learning_rate = 1e-5
momentum = .9

l_in = nntools.layers.InputLayer(shape=(1, X_val[0].shape[1]))
l_recurrent_1 = rnntools.LSTMLayer(l_in, num_units=156)
l_recurrent_2 = rnntools.LSTMLayer(l_recurrent_1, num_units=300)
l_recurrent_3 = rnntools.LSTMLayer(l_recurrent_2, num_units=102)
nonlinearity = nntools.nonlinearities.softmax
l_out = nntools.layers.DenseLayer(l_recurrent_3, num_units=y_val[0].shape[1],
                                  nonlinearity=nonlinearity)

# Cost function is mean squared error
input = T.matrix('input')
target_output = T.matrix('target_output')
cost = nntools.objectives.crossentropy(l_out.get_output(input), target_output)
# Use SGD for training
all_params = nntools.layers.get_all_params(l_out)
print 'Computing updates...'
updates = nntools.updates.momentum(cost, all_params, learning_rate, momentum)
print 'Compiling functions...'
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output], cost, updates=updates)
y_pred = theano.function([input], l_out.get_output(input))
compute_cost = theano.function([input, target_output], cost)

print 'Training...'
# Train the net
for epoch in range(n_epochs):
    start_time = time.time()
    for sequence, labels in zip(X_train, y_train):
        train(sequence.astype(theano.config.floatX),
              labels.astype(theano.config.floatX))
    end_time = time.time()
    cost_val = sum([compute_cost(X_val_n, y_val_n) for X_val_n, y_val_n
                    in zip(X_val, y_val)])
    y_val_pred = [y_pred(X_val_n) for X_val_n in X_val]
    error = np.mean(np.argmax(np.vstack(y_val), axis=0)
                    != np.argmax(np.vstack(y_val_pred), axis=0))
    print "Epoch {} took {}, cost = {}, error = {}".format(
        epoch, end_time - start_time, cost_val, error)
