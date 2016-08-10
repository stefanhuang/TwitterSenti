"""
Pipeline Runner, a module to wrap the tokeniser, feature extractor, and Classifier trainer
to perform training and testing on given training and test set specifically for task A and B
This runner can then be compiled and import to iPython notebook, where the tokeniser, 
feature extractor, and Classifier trainer will be writen, and use this module to test and help
tweaking the performance.
"""

import nltk
import csv
# from nltk.classify.util import apply_features
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from collections import Counter
import time
import functools
import numpy as np
import linecache

twitter_train_A_path = "Exercise2_data/twitter-train-cleansed-A.tsv"
twitter_train_B_path = "Exercise2_data/twitter-train-cleansed-B.tsv"

twitter_dev_gold_A_path = "Exercise2_data/twitter-dev-gold-A.tsv"
twitter_dev_gold_B_path = "Exercise2_data/twitter-dev-gold-B.tsv"

twitter_test_A_path = "Exercise2_data/twitter-test-A.tsv"
twitter_test_B_path = "Exercise2_data/twitter-test-B.tsv"

LABELS = {'positive','neutral','negative'}

# read a task B training file and return the training set (feature,label)
def getTrainingSetB(filepath, tokeniser):
    tokenlist = []
    labellist = []

    with open(filepath) as tsvfile:
        for aline in tsvfile:
            line = aline.strip().split('\t') # remove whitespace at start and end and split

            label = line[2]
            tokens = tokeniser(line[3])

            tokenlist.append(tokens)
            labellist.append(label)
            
    return { 'tokens':tokenlist, 'labels':labellist }

# read a task A training file and return the training set (feature,label)
def getTrainingSetA(filepath, tokeniser):
    tokenlist = []
    labellist = []

    with open(filepath) as tsvfile:
        for aline in tsvfile:
            line = aline.strip().split('\t')

            begin = int(line[2])
            end = int(line[3])
            label = line[4]
            tokens = tokeniser(line[5], begin=begin, end=end)

            tokenlist.append(tokens)
            labellist.append(label)

    return { 'tokens':tokenlist, 'labels':labellist }

# get the performance of the classifier with the given test set (set of features and label)
# the test set should contian marked labels
def getPerformance_on_testset(gold_set, pipeline, labels=LABELS):
    reference = gold_set['labels']
    test = list(pipeline.predict(gold_set['tokens']))

    cm = nltk.ConfusionMatrix(reference, test)
    print(cm.pretty_format(sort_by_count=False))
    
    # use helper methods to get macro f-measure
    mfs = get_Final_Score(labels,cm)
    return mfs, reference, test

# helper return the macro average F1 for the positive and negative classes.
def get_Final_Score(labels, cm):
    # helper function to get f score of all classes
    def getF_scores(labels, cm):
        TP = Counter()
        FN = Counter()
        FP = Counter()

        for i in labels:
            for j in labels:
                if i == j:
                    TP[i] += cm[i,j]
                else:
                    FN[i] += cm[i,j]
                    FP[j] += cm[i,j]

        f_measures = {}
        for i in labels:
            if TP[i] == 0:
                f_measure = 0
            else:
                precision = TP[i] / float(TP[i]+FP[i])
                recall = TP[i] / float(TP[i]+FN[i])
                f_measure = 2 * (precision * recall) / float(precision + recall)
            f_measures[i] = f_measure
            print ('%s f-measure: %f' % (i, f_measure))
        return f_measures
    
    # using the above helper function to get a list of f measures 
    # and calculate macro f-measure
    f_measures = getF_scores(labels, cm)
    mfs = float( f_measures['positive'] + f_measures['negative'])/2 
    
    print ( 'Macro f-measure: %f' % mfs )
    return mfs

# pipeline to perform 10 fold cross validation with a training set
# with given extract feature function, classifier training function
def TenFoldCV(pipeline, training_set):
    folds = cross_validation.KFold(len(training_set['labels']), n_folds=10, shuffle=False, random_state=None)

    gold = []
    result = []
    for train_idx, test_idx in folds:
        # train on indexes belong to the training fold
        # kclassifier = classifier_train_func([ training_set[idx] for idx in train_idx])
        train_text = [ training_set['tokens'][idx] for idx in train_idx]
        train_label = [ training_set['labels'][idx] for idx in train_idx]
        pipeline = pipeline.fit(train_text, train_label)

        # test on indexes belong to the testing fold
        kgold = [ training_set['labels'][idx] for idx in test_idx ]    # labels from test fold
        kresult = pipeline.predict([ training_set['tokens'][idx] for idx in test_idx  ]) # classify features
        gold.extend(kgold)
        result.extend(list(kresult))
        
        # print fold performance for tuning
        print ('test range: {} accuracy: {}'.format([test_idx[0], test_idx[-1]] , accuracy_score(kgold, kresult)) )
    
    # make confusion matrix from the whole cross validation result
    cm = nltk.ConfusionMatrix(gold, result)
    print(cm.pretty_format(sort_by_count=False))
    
    labels = {'neutral','positive','negative'}
    
    # use helper methods to get macro f-measure
    mfs = get_Final_Score(labels,cm)
    
    # return gold and result for further examination for tuning
    return mfs, gold, result

# helper method to run 10 fold cross validation and test on dev set 
# for given training and gold set
def runAll(pipeline, training_set, gold_set):
    # 10 fold validation
    print("\n>>>Start 10 fold validation:")
    tf_mfs, tf_gold, tf_result = TenFoldCV(pipeline, training_set)

    # train the classifier on the whole training set and test on dev set 
    print("\n>>>Start applying pipeline to train classifier on whole training set and test on dev set:")
    start = time.clock()
    # pipeline = classifier_train_func(training_set)
    pipeline = pipeline.fit(training_set['tokens'], training_set['labels'])
    end = time.clock()
    print ("~~ Training classifier took (sec):")
    print (end-start)
    
    # get performance with dev gold
    dev_mfs, dev_gold, dev_result = getPerformance_on_testset(gold_set, pipeline)

    # return all the results for inspection
    return  pipeline, tf_mfs, tf_gold, tf_result, dev_mfs, dev_gold, dev_result

# pipeline to perform 10 fold cross validation and test on dev set for task A and B
def runAllTaskA(pipeline, tokeniser):
    # get trainning set for task A 
    training_set = getTrainingSetA(twitter_train_A_path, tokeniser)
    gold_set = getTrainingSetA(twitter_dev_gold_A_path, tokeniser)

    # return all the results for inspection
    return  runAll(pipeline, training_set, gold_set)


# pipeline to perform 10 fold cross validation and test on dev set for task A and B
def runAllTaskB(pipeline, tokeniser):
    # get trainning set for task B
    training_set = getTrainingSetB(twitter_train_B_path, tokeniser)
    gold_set = getTrainingSetB(twitter_dev_gold_B_path, tokeniser)
    
    # return all the results for inspection
    return  runAll(pipeline, training_set, gold_set)

# inspect the give line numbers (in a list) in the input file, 
# for tweaking and debugging Task B
def inspectTweetsB(filepath, linenumbers):
    tweets = []
    for linenumber in linenumbers:
        line = linecache.getline(filepath, linenumber+1).split('\t') # we count from 0 for lines
        tweets.append((linenumber, line[2], line[3]))
    return tweets

# inspect the give line numbers (in a list) in the input file, for tweaking and debugging
# for tweaking and debugging Task A, 
# need the tokeniser because we need to extract instance of a word or phrase
def inspectTweetsA(filepath, linenumbers, tokeniser):
    tweets = []
    for linenumber in linenumbers:
        line = linecache.getline(filepath, linenumber+1).split('\t')
        begin = int(line[2])
        end = int(line[3])
        label = line[4]
        tokens = tokeniser(line[5], begin=begin, end=end)
        tweets.append((linenumber, label, tokens))
    return tweets  
