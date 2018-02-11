import os
import tensorflow as tf
import time
import numpy as np

from utils import DKT
from load_data import DKTData


"""
Assignable variables:
num_runs: int
num_epochs: int
keep_prob: float
batch_size: int
hidden_layer_structure: tuple
data_dir: str
train_file_name: str
test_file_name: str
ckpt_save_dir: str
"""
import argparse
parser = argparse.ArgumentParser()

# network configuration
parser.add_argument("-hl", "--hidden_layer_structure", default=[200, ], nargs='*', type=int,
                    help="The hidden layer structure in the RNN. If there is 2 hidden layers with first layer "
                         "of 200 and second layer of 50. Type in '-hl 200 50'")
parser.add_argument("-cell", "--rnn_cell", default='LSTM', choices=['LSTM', 'GRU', 'BasicRNN', 'LayerNormBasicLSTM'],
                    help='Specify the rnn cell used in the graph.')
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2,
                    help="The learning rate when training the model.")
parser.add_argument("-kp", "--keep_prob", type=float, default=0.5,
                    help="Keep probability when training the network.")
parser.add_argument("-mgn", "--max_grad_norm", type=float, default=5.0,
                    help="The maximum gradient norm allowed when clipping.")
parser.add_argument("-lw1", "--lambda_w1", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l1-norm.")
parser.add_argument("-lw2", "--lambda_w2", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l2-norm.")
parser.add_argument("-lo", "--lambda_o", type=float, default=0.00,
                    help="The lambda coefficient for the regularization objective.")
# training configuration
parser.add_argument("--num_runs", type=int, default=5,
                    help="Number of runs to repeat the experiment.")
parser.add_argument("--num_epochs", type=int, default=500,
                    help="Maximum number of epochs to train the network.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="The mini-batch size used when training the network.")
# data file configuration
parser.add_argument('--data_dir', type=str, default='./data/',
                    help="the data directory, default as './data/")
parser.add_argument('--train_file', type=str, default='skill_id_train.csv',
                    help="train data file, default as 'skill_id_train.csv'.")
parser.add_argument('--test_file', type=str, default='skill_id_test.csv',
                    help="train data file, default as 'skill_id_test.csv'.")
parser.add_argument("-csd", "--ckpt_save_dir", type=str, default=None,
                    help="checkpoint save directory")
parser.add_argument('--dataset', type=str, default='a2009')
args = parser.parse_args()

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

dataset = args.dataset
if dataset == 'a2009u':
    train_path = './data/assist2009_updated/assist2009_updated_train.csv'
    test_path = './data/assist2009_updated/assist2009_updated_test.csv'
    save_dir_prefix = './a2009u/'
elif dataset == 'a2015':
    train_path = './data/assist2015/assist2015_train.csv'
    test_path = './data/assist2015/assist2015_test.csv'
    save_dir_prefix = './a2015/'
elif dataset == 'synthetic':
    train_path = './data/synthetic/naive_c5_q50_s4000_v1_train.csv'
    test_path = './data/synthetic/naive_c5_q50_s4000_v1_test.csv'
    save_dir_prefix = './synthetic/'
elif dataset == 'statics':
    train_path = './data/STATICS/STATICS_train.csv'
    test_path = './data/STATICS/STATICS_test.csv'
    save_dir_prefix = './STATICS/'
elif dataset =='assistment_challenge':
    train_path = './data/assistment_challenge/assistment_challenge_train.csv'
    test_path = './data/assistment_challenge/assistment_challenge_test.csv'
    save_dir_prefix = './assistment_challenge/'
elif dataset == 'toy':
    train_path = './data/toy_data_train.csv'
    test_path = './data/toy_data_test.csv'
    save_dir_prefix = './toy/'
elif dataset == 'a2009':
    train_path = './data/skill_id_train.csv'
    test_path = './data/skill_id_test.csv'
    save_dir_prefix = './a2009/'


network_config = {}
network_config['batch_size'] = args.batch_size
network_config['hidden_layer_structure'] = list(args.hidden_layer_structure)
network_config['learning_rate'] = args.learning_rate
network_config['keep_prob'] = args.keep_prob
network_config['rnn_cell'] = rnn_cells[args.rnn_cell]
network_config['lambda_w1'] = args.lambda_w1
network_config['lambda_w2'] = args.lambda_w2
network_config['lambda_o'] = args.lambda_o

num_runs = args.num_runs
num_epochs = args.num_epochs
batch_size = args.batch_size
keep_prob = args.keep_prob

ckpt_save_dir = args.ckpt_save_dir
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_test, num_problems, network_config,
              save_dir_prefix=save_dir_prefix,
              num_runs=num_runs, num_epochs=num_epochs,
              keep_prob=keep_prob, logging=True, save=True)

    # run optimization of the created model
    dkt.model.build_graph()
    dkt.run_optimization()
    # close the session
    sess.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("program run for: {0}s".format(end_time - start_time))
