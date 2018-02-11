import os
import csv
import numpy as np
from sklearn.utils import shuffle


def pad(data, target_length, target_value=0):
    return np.pad(data, (0, target_length - len(data)), 'constant', constant_values=target_value)


def one_hot(indices, depth):
    encoding = np.concatenate((np.eye(depth), [np.zeros(depth)]))
    return encoding[indices]


class OriginalInputProcessor(object):
    def process_problems_and_corrects(self, problem_seqs, correct_seqs, num_problems, is_train=True):
        """
        This function aims to process the problem sequence and the correct sequence into a DKT feedable X and y.
        :param problem_seqs: it is in shape [batch_size, None]
        :param correct_seqs: it is the same shape as problem_seqs
        :return:
        """
        # pad the sequence with the maximum sequence length
        max_seq_length = max([len(problem) for problem in problem_seqs])
        problem_seqs_pad = np.array([pad(problem, max_seq_length, target_value=-1) for problem in problem_seqs])
        correct_seqs_pad = np.array([pad(correct, max_seq_length, target_value=-1) for correct in correct_seqs])

        # find the correct seqs matrix as the following way:
        # Let problem_seq = [1,3,2,-1,-1] as a and correct_seq = [1,0,1,-1,-1] as b, which are padded already
        # First, find the element-wise multiplication of a*b*b = [1,0,2,-1,-1]
        # Then, for any values 0, assign it to -1 in the vector = [1,-1,2,-1,-1] as c
        # Such that when we one hot encoding the vector c, it will results a zero vector
        temp = problem_seqs_pad * correct_seqs_pad * correct_seqs_pad  # temp is c in the comment.
        temp[temp == 0] = -1
        correct_seqs_pad = temp

        # one hot encode the information
        problem_seqs_oh = one_hot(problem_seqs_pad, depth=num_problems)
        correct_seqs_oh = one_hot(correct_seqs_pad, depth=num_problems)

        # slice out the x and y
        if is_train:
            x_problem_seqs = problem_seqs_oh[:, :-1]
            x_correct_seqs = correct_seqs_oh[:, :-1]
            y_problem_seqs = problem_seqs_oh[:, 1:]
            y_correct_seqs = correct_seqs_oh[:, 1:]
        else:
            x_problem_seqs = problem_seqs_oh[:, :]
            x_correct_seqs = correct_seqs_oh[:, :]
            y_problem_seqs = problem_seqs_oh[:, :]
            y_correct_seqs = correct_seqs_oh[:, :]

        X = np.concatenate((x_problem_seqs, x_correct_seqs), axis=2)

        result = (X, y_problem_seqs, y_correct_seqs)
        return result


class BatchGenerator:
    """
    Generate batch for DKT model
    """

    def __init__(self, problem_seqs, correct_seqs, num_problems, batch_size, input_processor=OriginalInputProcessor(),
                 **kwargs):
        self.cursor = 0  # point to the current batch index
        self.problem_seqs = problem_seqs
        self.correct_seqs = correct_seqs
        self.batch_size = batch_size
        self.num_problems = num_problems
        self.num_samples = len(problem_seqs)
        self.num_batches = len(problem_seqs) // batch_size + 1
        self.input_processor = input_processor
        self._current_batch = None

    def next_batch(self, is_train=True):
        start_idx = self.cursor * self.batch_size
        end_idx = min((self.cursor + 1) * self.batch_size, self.num_samples)
        problem_seqs = self.problem_seqs[start_idx:end_idx]
        correct_seqs = self.correct_seqs[start_idx:end_idx]

        # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
        self._current_batch = self.input_processor.process_problems_and_corrects(problem_seqs,
                                                                                 correct_seqs,
                                                                                 self.num_problems,
                                                                                 is_train=is_train)
        self._update_cursor()
        return self._current_batch

    @property
    def current_batch(self):
        if self._current_batch is None:
            print("Current batch is None.")
        return None

    def _update_cursor(self):
        self.cursor = (self.cursor + 1) % self.num_batches

    def reset_cursor(self):
        self.cursor = 0

    def shuffle(self):
        self.problem_seqs, self.correct_seqs = shuffle(self.problem_seqs, self.correct_seqs, random_state=42)


def read_data_from_csv(filename):
    # read the csv file
    rows = []
    with open(filename, 'r') as f:
        print("Reading {0}".format(filename))
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
        print("{0} lines was read".format(len(rows)))

    # tuples stores the student answering sequence as
    # ([num_problems_answered], [problem_ids], [is_corrects])
    max_seq_length = 0
    num_problems = 0
    tuples = []
    for i in range(0, len(rows), 3):
        # numbers of problem a student answered
        seq_length = int(rows[i][0])

        # only keep student with at least 3 records.
        if seq_length < 3:
            continue

        problem_seq = rows[i + 1]
        correct_seq = rows[i + 2]

        invalid_ids_loc = [i for i, pid in enumerate(problem_seq) if pid == '']
        for invalid_loc in invalid_ids_loc:
            del problem_seq[invalid_loc]
            del correct_seq[invalid_loc]

        # convert the sequence from string to int.
        problem_seq = list(map(int, problem_seq))
        correct_seq = list(map(int, correct_seq))

        tup = (seq_length, problem_seq, correct_seq)
        tuples.append(tup)

        if max_seq_length < seq_length:
            max_seq_length = seq_length

        pid = max(int(pid) for pid in problem_seq if pid != '')
        if num_problems < pid:
            num_problems = pid
    # add 1 to num_problems because 0 is in the pid
    num_problems += 1

    print("max_num_problems_answered:", max_seq_length)
    print("num_problems:", num_problems)
    print("The number of students is {0}".format(len(tuples)))
    print("Finish reading data.")

    return tuples, num_problems, max_seq_length


class DKTData:
    def __init__(self, train_path, test_path, batch_size=32):
        self.students_train, num_problems_train, max_seq_length_train = read_data_from_csv(train_path)
        self.students_test, num_problems_test, max_seq_length_test = read_data_from_csv(test_path)
        self.num_problems = max(num_problems_test, num_problems_train)
        self.max_seq_length = max(max_seq_length_train, max_seq_length_test)

        problem_seqs = [student[1] for student in self.students_train]
        correct_seqs = [student[2] for student in self.students_train]
        self.train = BatchGenerator(problem_seqs, correct_seqs, self.num_problems, batch_size)

        problem_seqs = [student[1] for student in self.students_test]
        correct_seqs = [student[2] for student in self.students_test]
        self.test = BatchGenerator(problem_seqs, correct_seqs, self.num_problems, batch_size)
