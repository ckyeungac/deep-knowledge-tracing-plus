# DKT+
This is the repository for the code in the paper *Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization* ([ACM](https://dl.acm.org/citation.cfm?id=3231647), [pdf](https://arxiv.org/pdf/1806.02180.pdf))

If you find this repository useful, please cite
```
@inproceedings{LS2018_Yeung_DKTP,
  title={Addressing two problems in deep knowledge tracing via prediction-consistent regularization},
  author={Yeung, Chun Kit and Yeung, Dit Yan},
  year={2018},
  booktitle = {{Proceedings of the 5th ACM Conference on Learning @ Scale}},
  pages = {5:1--5:10},
  publisher = {ACM},
}
```

## Abstact
Knowledge tracing is one of the key research areas for empowering personalized education. It is a task to model students' mastery level of a knowledge component (KC) based on their historical learning trajectories. In recent years, a recurrent neural network model called deep knowledge tracing (DKT) has been proposed to handle the knowledge tracing task and literature has shown that DKT generally outperforms traditional methods. However, through our extensive experimentation, we have noticed two major problems in the DKT model. The first problem is that the model fails to reconstruct the observed input. As a result, even when a student performs well on a KC, the prediction of that KC's mastery level decreases instead, and vice versa. Second, the predicted performance across time-steps is not consistent. This is undesirable and unreasonable because student's performance is expected to transit gradually over time. To address these problems, we introduce regularization terms that correspond to \emph{reconstruction} and \textit{waviness} to the loss function of the original DKT model to enhance the consistency in prediction. Experiments show that the regularized loss function effectively alleviates the two problems without degrading the original task of DKT.

## Requirements
I have used tensorflow to develop the deep knowledge tracing model, and the following is the packages I used:
```
tensorflow==1.2.0 (or tensorflow-gpu==1.3.0)
scikit-learn==0.18.1
scipy==0.19.0
numpy==1.13.3
```

The packages used for the visualization of the student knowledge state are
```
seaborn
matplotlib
```

## Data Format
The first line the number of exercises a student attempted. The second line is the exercise tag sequence. The third line is the response sequence.
```
15
1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
```

## Program Usage
### Run the experiment
```
python main.py
```

### Detail hyperparameter for the program
```
usage: main.py [-h]
               [-hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]]]
               [-cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}]
               [-lr LEARNING_RATE] [-kp KEEP_PROB] [-mgn MAX_GRAD_NORM]
               [-lw1 LAMBDA_W1] [-lw2 LAMBDA_W2] [-lo LAMBDA_O]
               [--num_runs NUM_RUNS] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE] [--data_dir DATA_DIR]
               [--train_file TRAIN_FILE] [--test_file TEST_FILE]
               [-csd CKPT_SAVE_DIR] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  -hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]], --hidden_layer_structure [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]]
                        The hidden layer structure in the RNN. If there is 2
                        hidden layers with first layer of 200 and second layer
                        of 50. Type in '-hl 200 50'
  -cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}, --rnn_cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}
                        Specify the rnn cell used in the graph.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate when training the model.
  -kp KEEP_PROB, --keep_prob KEEP_PROB
                        Keep probability when training the network.
  -mgn MAX_GRAD_NORM, --max_grad_norm MAX_GRAD_NORM
                        The maximum gradient norm allowed when clipping.
  -lw1 LAMBDA_W1, --lambda_w1 LAMBDA_W1
                        The lambda coefficient for the regularization waviness
                        with l1-norm.
  -lw2 LAMBDA_W2, --lambda_w2 LAMBDA_W2
                        The lambda coefficient for the regularization waviness
                        with l2-norm.
  -lo LAMBDA_O, --lambda_o LAMBDA_O
                        The lambda coefficient for the regularization
                        objective.
  --num_runs NUM_RUNS   Number of runs to repeat the experiment.
  --num_epochs NUM_EPOCHS
                        Maximum number of epochs to train the network.
  --batch_size BATCH_SIZE
                        The mini-batch size used when training the network.
  --data_dir DATA_DIR   the data directory, default as './data/
  --train_file TRAIN_FILE
                        train data file, default as 'skill_id_train.csv'.
  --test_file TEST_FILE
                        train data file, default as 'skill_id_test.csv'.
  -csd CKPT_SAVE_DIR, --ckpt_save_dir CKPT_SAVE_DIR
                        checkpoint save directory
  --dataset DATASET
```
