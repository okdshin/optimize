#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset
This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""

import argparse
import os

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers

import train_mnist

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=2,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot using model '
                             'and state files in the specified directory')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    print(args)

    #device = chainer.get_device(args.device)

    #print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = L.Classifier(train_mnist.MLP(args.unit, 10))
    #model.to_device(device)
    #device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(args.lr, args.momentum)
    optimizer.setup(model)

    if args.resume is not None:
        # Resume from a snapshot
        resume = args.resume
        if os.path.exists(resume):
            serializers.load_npz(os.path.join(resume, 'mlp.model'), model)
            serializers.load_npz(os.path.join(resume, 'mlp.state'), optimizer)
        else:
            raise ValueError(
                '`args.resume` ("{}") is specified,'
                ' but it does not exist'.format(resume)
            )

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x, t = convert.concat_examples(batch)#, device)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.array) * len(t)
        sum_accuracy += float(model.accuracy.array) * len(t)

        if train_iter.is_new_epoch:
            print('epoch: {}'.format(train_iter.epoch))
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            test_sum_accuracy = 0
            test_sum_loss = 0
            # Enable evaluation mode.
            with configuration.using_config('train', False):
                # This is optional but can reduce computational overhead.
                with chainer.using_config('enable_backprop', False):
                    for batch in test_iter:
                        x, t = convert.concat_examples(batch)#, device)
                        loss = model(x, t)
                        test_sum_loss += float(loss.array) * len(t)
                        test_sum_accuracy += float(
                            model.accuracy.array) * len(t)

            test_iter.reset()
            print('test mean  loss: {}, accuracy: {}'.format(
                test_sum_loss / test_count, test_sum_accuracy / test_count))
            print('OPTUNA_STEP', train_iter.epoch, test_sum_loss)
            sum_accuracy = 0
            sum_loss = 0
    print('OPTUNA_FINISHED', test_sum_loss)

    # Save the model and the optimizer
    out = args.out
    if not os.path.isdir(out):
        os.makedirs(out)
    print('save the model')
    serializers.save_npz(os.path.join(out, 'mlp.model'), model)
    print('save the optimizer')
    serializers.save_npz(os.path.join(out, 'mlp.state'), optimizer)


if __name__ == '__main__':
    main()
