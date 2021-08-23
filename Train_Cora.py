import argparse
import os
import tensorflow as tf
from utils.classifier import Classifier
from Network.Trainer import Trainer
from utils import process
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.fftpack import fft
import winsound
import time

def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--dataset', nargs='?', default='cora', help='Input dataset')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate. Default is 0.001.')
    parser.add_argument('--dlr', type=float, default=3e-5, help='D Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--lambda-', default=0.1, type=float, help='Parameter controlling the contribution of edge '
                                                                  'reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=5.0, type=float, help='gradient clipping')
    return parser.parse_args()


def main(args):
    """
    Pipeline for Graph Attention Auto-encoder.
    """
    G, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    print('Graph-dimension：' + str(G.shape))
    print('Content-dimension：' + str(X.shape))
    Label = np.array([np.argmax(l) for l in Y])
    print('Label-dimension：' + str(Label.shape))
    # add feature dimension size to the beginning of hidden_dims
    feature_dim = X.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims
    print('hidden-1-dimension：' + str(args.hidden_dims))

    # prepare the data
    G_tf, S, R = process.prepare_graph_data(G)
    # PreTrain the Model
    # fin = False
    trainer = Trainer(args)
    _ = trainer.assign(G_tf, X, S, R)
    # trainer(G_tf, X, S, R, Label, fin)
    # Fintune the Model
    fin = True
    trainer(G_tf, X, S, R, Label, fin)


if __name__ == "__main__":

    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
