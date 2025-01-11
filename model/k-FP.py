import os
import time
import csv
import sys
import dill
import random
import argparse
import operator
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score     # CHANGE: new verion of library
from sklearn import metrics
from sklearn import tree
import sklearn.metrics as skm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger_config import *

# re-seed the generator
#np.random.seed(1234)

parser = argparse.ArgumentParser(description='k-FP benchmarks')
parser.add_argument('--rule-name', type=str, default="rule_name", help='Name of the defense rule')
parser.add_argument('--pkl-path', type=str, default="/path/to/train.pkl", help='Path to the X & Y features file')
parser.add_argument('--model-result', type=str, default="/path/to/result.txt", help='Path to the model accuracy log')
parser.add_argument('--log-path', type=str, default="/path/to/log_folder", help='Root folder for log data')
args = parser.parse_args()

defense_rule_name = args.rule_name
model_result = args.model_result
path_to_dict = args.pkl_path

### Parameters ###
# Number of sites, number of instances per site, number of (alexa/hs) monitored training instances per site, Number of trees for RF etc.
num_Trees = 1000

alexa_sites = 258                       # train:test:validate = 8:1:1
alexa_instances = 800
alexa_train_inst = 640

mon_train_inst = alexa_train_inst
mon_test_inst = alexa_instances - mon_train_inst
n_jobs = 128

logger = setup_logger(args.log_path)

############ Feeder functions ############
def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):       # CHANGE: xrange to range
        yield l[i:i+n]


def checkequal(lst):
    return lst[1:] == lst[:-1]


############ Non-Feeder functions ########
def mon_train_test_references(path_to_dict):
    """ Prepare monitored data in to training and test sets. """

    fileObject1 = open(path_to_dict, 'rb')      # CHANGE: 'r' to 'rb'
    dic = dill.load(fileObject1)

    split_data = list(chunks(dic['alexa_feature'], alexa_instances))
    split_target = list(chunks(dic['alexa_label'], alexa_instances))

    training_data = []
    training_label = []
    test_data = []
    test_label = []

    for i in range(len(split_data)):
        temp = list(zip(split_data[i], split_target[i]))        # CHANGE: convert zip to list
        random.shuffle(temp)
        data, label = zip(*temp)
        training_data.extend(data[:mon_train_inst])
        training_label.extend(label[:mon_train_inst])
        test_data.extend(data[mon_train_inst:])
        test_label.extend(label[mon_train_inst:])

    flat_train_data = []
    flat_test_data = []

    # CHANGE: tuple () to list []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, [])))
    for te in test_data:
        flat_test_data.append(list(sum(te, [])))

    # CHANGE: covert zip to list
    training_features =  list(zip(flat_train_data, training_label))
    test_features =  list(zip(flat_test_data, test_label))

    return training_features, test_features


def RF_closedworld(defense_rule_name, model_result, path_to_dict, n_jobs = n_jobs):
    '''Closed world RF classification of data -- only uses sk.learn classification - does not do additional k-nn.'''

    training, test = mon_train_test_references(path_to_dict)
    tr_data, tr_label1 = zip(*training)

    # CHANGE: tuple to list
    tr_label = list(zip(*tr_label1))[0]
    te_data, te_label1 = zip(*test)
    te_label = list(zip(*te_label1))[0]

    logger.info("Running k-FP model...")

    model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=num_Trees, oob_score=True, verbose=1)    # CHANGE: added verbose
    model.fit(tr_data, tr_label)
    
    logger.info(f"RF accuracy = {model.score(te_data, te_label):.4f}")
    logger.info(f"Feature importance scores: {model.feature_importances_}")

    scores = cross_val_score(model, np.array(tr_data), np.array(tr_label))
    logger.info(f"cross_val_score = {scores.mean():.4f}")
    logger.info(f"OOB score = {model.oob_score_:.4f}")


if __name__ == "__main__":
    RF_closedworld(defense_rule_name, model_result, path_to_dict)