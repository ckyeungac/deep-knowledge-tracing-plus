import os
import tensorflow as tf
import time
import numpy as np
from utils import DKT
from load_data import DKTData
from math import log


import argparse
parser = argparse.ArgumentParser()
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
# random search hyperparameter
parser.add_argument("-hsz_mean", "--hyper_state_size_mean", type=float, default=200.0,
                    help="The target state size of mean of lognormal of generating hidden layer state size.")
parser.add_argument("-hsz_std", "--hyper_state_size_std", type=float, default=0.1,
                    help="The target state size of standard deviation of lognormal of "
                         "generating hidden layer state size.")

# hyper-parameter
parser.add_argument("-lw1", "--lambda_w1_values", type=float, default=[0.00,], nargs='*',
                    help="The lambda coefficient for the regularization waviness with l1-norm.")
parser.add_argument("-lw2", "--lambda_w2_values", type=float, default=[0.00,], nargs='*',
                    help="The lambda coefficient for the regularization waviness with l2-norm.")
parser.add_argument("-lo", "--lambda_o_values", type=float, default=[0.00,], nargs='*',
                    help="The lambda coefficient for the regularization objective.")

args = parser.parse_args()


train_path = os.path.join(args.data_dir, args.train_file)
test_path = os.path.join(args.data_dir, args.test_file)
num_runs = args.num_runs
num_epochs = args.num_epochs
batch_size = args.batch_size

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

def loguniform(low=0.0, high=1.0, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

def generate_hyperparameter():
    rnn_cell = 'LSTM'
    # num_layers = np.random.choice([1,2,3], p=[0.5, 0.4, 0.1])
    # hidden_layer_structure = []
    # for i in range(1, num_layers+1):
    #     state_size = np.random.lognormal(mean=log(args.hyper_state_size_mean),
    #                                          sigma=args.hyper_state_size_std)
    #     state_size = int(state_size // i) # int operation is required.
    #     hidden_layer_structure.append(state_size)
    hidden_layer_structure = [200,]
    # learning_rate = loguniform(low=0.001, high=0.01)
    learning_rate = 0.01
    # keep_prob = np.random.uniform(low=0.3, high=0.5)
    keep_prob = 0.5
    embedding_size = int(np.random.uniform(low=20, high=400))
    lambda_o = np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25], p=[.2, .2, .2, .2, .2])
    lambda_w = np.random.choice([0.001, 0.003, 0.01, 0.03, 0.1], p=[.2, .2, .2, .2, .2])

    network_config = {}
    network_config['batch_size'] = 32
    network_config['rnn_cell'] = rnn_cells[rnn_cell]
    network_config['hidden_layer_structure'] = hidden_layer_structure
    network_config['learning_rate'] = learning_rate
    network_config['keep_prob'] = keep_prob
    network_config['embedding_size'] = embedding_size
    network_config['lambda_o'] = lambda_o
    network_config['lambda_w1'] = lambda_w
    return network_config


def main(num_search=5):
    best_avg_auc = 0.0
    best_network_config = None
    best_dkt = None
    network_config = generate_hyperparameter()
    for lambda_o in args.lambda_o_values:
        for lambda_w1 in args.lambda_w1_values:
            for lambda_w2 in args.lambda_w2_values:
                # setting network parameter
                network_config['lambda_o'] = lambda_o
                network_config['lambda_w1'] = lambda_w1
                network_config['lambda_w2'] = lambda_w2

                keep_prob = network_config['keep_prob']
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)

                data = DKTData(train_path, test_path, batch_size=batch_size)
                data_train = data.train
                data_test = data.test
                num_problems = data.num_problems

                dkt = DKT(sess, data_train, data_test, num_problems, network_config,
                          ckpt_save_dir=None,
                          num_runs=num_runs, num_epochs=num_epochs,
                          keep_prob=keep_prob, logging=True, save=True)

                # run optimization of the created model
                dkt.model.build_graph()
                avg_auc = dkt.run_optimization()
                if avg_auc > best_avg_auc:
                    best_avg_auc = avg_auc
                    best_network_config = network_config
                    best_dkt = dkt

                # close the session
                sess.close()
                tf.reset_default_graph()
        
    print("Best avg. auc:", best_avg_auc)
    print("Best network config:", best_network_config)
    print("Best dkt log path:", best_dkt.log_file_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("program run for: {0}s".format(end_time - start_time))