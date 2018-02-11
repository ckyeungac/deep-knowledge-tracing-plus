import numpy as np
import tensorflow as tf


def length(sequence):
    """
    This function return the sequence length of each x in the batch.
    :param sequence: the batch sequence of shape [batch_size, num_steps, feature_size]
    :return length: A tensor of shape [batch_size]
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    seq_length = tf.reduce_sum(used, 1)
    seq_length = tf.cast(seq_length, tf.int32)
    return seq_length


# reference:
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
# https://github.com/davidoj/deepknowledgetracingTF/blob/master/model.py
class Model(object):
    def __init__(self, num_problems,
                 hidden_layer_structure=(200,),
                 batch_size=32,
                 rnn_cell=tf.contrib.rnn.LSTMCell,
                 learning_rate=0.01,
                 max_grad_norm=5.0,
                 lambda_w1 = 0.0,
                 lambda_w2 = 0.0,
                 lambda_o = 0.0,
                 **kwargs):
        # dataset-dependent attributes
        self.num_problems = num_problems
        self.hidden_layer_structure = hidden_layer_structure
        self.batch_size = batch_size
        self.rnn_cell = rnn_cell
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lambda_w1 = lambda_w1 # regularization parameter for waviness for l1-norm
        self.lambda_w2 = lambda_w2 # regularization parameter for waviness for l1-norm
        self.lambda_o = lambda_o # regularization parameter for objective function

    def _create_placeholder(self):
        print("Creating placeholder...")
        num_problems = self.num_problems

        # placeholder
        self.X = tf.placeholder(tf.float32, [None, None, 2 * num_problems], name='X')
        self.y_seq = tf.placeholder(tf.float32, [None, None, num_problems], name='y_seq')
        self.y_corr = tf.placeholder(tf.float32, [None, None, num_problems], name='y_corr')
        self.keep_prob = tf.placeholder(tf.float32)
        self.hidden_layer_input = self.X
        self.seq_length = length(self.X)

    def _influence(self):
        print("Creating Loss...")
        hidden_layer_structure = self.hidden_layer_structure

        # Hidden Layer Construction
        self.hidden_layers_outputs = []
        self.hidden_layers_state = []
        hidden_layer_input = self.hidden_layer_input
        for i, layer_state_size in enumerate(hidden_layer_structure):
            variable_scope_name = "hidden_layer_{}".format(i)
            with tf.variable_scope(variable_scope_name, reuse=tf.get_variable_scope().reuse):
                cell = self.rnn_cell(num_units=layer_state_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    hidden_layer_input,
                    dtype=tf.float32,
                    sequence_length=self.seq_length
                )
            self.hidden_layers_outputs.append(outputs)
            self.hidden_layers_state.append(state)
            hidden_layer_input = outputs

    def _create_loss(self):
        print("Creating Loss...")
        last_layer_size = self.hidden_layer_structure[-1]
        last_layer_outputs = self.hidden_layers_outputs[-1]

        # Output Layer Construction
        with tf.variable_scope("output_layer", reuse=tf.get_variable_scope().reuse):
            W_yh = tf.get_variable("weights", shape=[last_layer_size, self.num_problems],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.num_problems)))
            b_yh = tf.get_variable("biases", shape=[self.num_problems, ],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.num_problems)))

            # Flatten the last layer output
            num_steps = tf.shape(last_layer_outputs)[1]
            self.outputs_flat = tf.reshape(last_layer_outputs, shape=[-1, last_layer_size])
            self.logits_flat = tf.matmul(self.outputs_flat, W_yh) + b_yh
            self.logits = tf.reshape(self.logits_flat, shape=[-1, num_steps, self.num_problems])
            self.preds = tf.sigmoid(self.logits)

            # self.preds_flat = tf.sigmoid(self.logits_flat)
            # y_seq_flat = tf.cast(tf.reshape(self.y_seq, [-1, self.num_problems]), dtype=tf.float32)
            # y_corr_flat = tf.cast(tf.reshape(self.y_corr, [-1, self.num_problems]), dtype=tf.float32)

            # Filter out the target indices as follow:
            # Get the indices where y_seq_flat are not equal to 0, where the indices
            # implies that a student has answered the question in the time step and
            # thereby exclude those time step that the student hasn't answered.
            target_indices = tf.where(tf.not_equal(self.y_seq, 0))

            self.target_logits = tf.gather_nd(self.logits, target_indices)
            self.target_preds = tf.gather_nd(self.preds, target_indices)  # needed to return AUC
            self.target_labels = tf.gather_nd(self.y_corr, target_indices)

            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_logits,
                                                                         labels=self.target_labels)

            self.loss = tf.reduce_mean(self.cross_entropy)


            # add current performance into consideration
            current_seq = self.X[:,:,:self.num_problems] # slice out the answering exercise
            current_corr = self.X[:,:,self.num_problems:]
            self.target_indices_current = tf.where(tf.not_equal(current_seq, 0))
            self.target_logits_current = tf.gather_nd(self.logits, self.target_indices_current)
            self.target_preds_current = tf.gather_nd(self.preds, self.target_indices_current) # needed to return AUC
            self.target_labels_current = tf.gather_nd(current_corr, self.target_indices_current)

            self.cross_entropy_current = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_logits_current,
                                                                                 labels=self.target_labels_current)
            self.loss += self.lambda_o * tf.reduce_mean(self.cross_entropy_current)


            # Regularize the model to smoothen the network.
            mask = length(self.y_seq)
            self.total_num_steps = tf.reduce_sum(tf.cast(mask, tf.float32))

            # l1-norm
            # waviness_norm_l1 = tf.norm(self.preds[:, 1:, :] - self.preds[:, :-1, :], ord=1)
            waviness_norm_l1 = tf.abs(self.preds[:, 1:, :] - self.preds[:, :-1, :])
            self.waviness_l1 = tf.reduce_sum(waviness_norm_l1) / self.total_num_steps / self.num_problems
            self.loss += self.lambda_w1 * self.waviness_l1

            # l2-norm
            # waviness_norm_l2 = tf.norm(self.preds[:, 1:, :] - self.preds[:, :-1, :], ord=2)
            waviness_norm_l2 = tf.square(self.preds[:, 1:, :] - self.preds[:, :-1, :])
            self.waviness_l2 = tf.reduce_sum(waviness_norm_l2) / self.total_num_steps / self.num_problems
            self.loss += self.lambda_w2 * self.waviness_l2


    def _create_optimizer(self):
        print('Create optimizer...')
        with tf.variable_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gvs = self.optimizer.compute_gradients(self.loss)
            clipped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(clipped_gvs)

    def _add_summary(self):
        pass


    def build_graph(self):
        self._create_placeholder()
        self._influence()
        self._create_loss()
        self._create_optimizer()
        self._add_summary()
        tf.get_variable_scope().reuse_variables() # better to add this to reuse the variable name.
