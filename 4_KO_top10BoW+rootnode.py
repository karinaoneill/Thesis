#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:32:24 2017

@author: karinaoneill
"""

from swda import CorpusReader
import csv


Want to create a CSV file named perfectmatchdatatop10.csv in which each utterance utt has its own row
with all the needed data headings and only utterances where tree is perfect match to be included.
Use swda_functions.py and swda.py for this. Save to pandas dataframe for later manipulation.
"""
"""
print ("Creating perfectmatchdatatop10.csv from swda corpus")

top10targets = list(['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc', 'bk'])

csvwriter = csv.writer(open('perfectmatchdatatop10.csv', 'wt'))
csvwriter.writerow(['swda_filename','ptb_basename','conversation_no','transcript_index','act_tag','caller',
                    'utterance_index','subutterance_index','text','pos','trees','ptb_treenumbers','damsl_act_tag',
                    'root_node','sent_length'])
corpus = CorpusReader('swda')    
for utt in corpus.iter_utterances(display_progress=True):
    if utt.tree_is_perfect_match() and utt.damsl_act_tag() in top10targets:
        csvwriter.writerow([utt.swda_filename, utt.ptb_basename, utt.conversation_no, utt.transcript_index,
                            utt.act_tag, utt.caller, utt.utterance_index, utt.subutterance_index, utt.text,
                            utt.pos, utt.trees, utt.ptb_treenumbers, utt.damsl_act_tag(), utt.trees[0].label(), len(utt.pos_words())])
        #changed .node to .label()

print ("perfectmatchdatatop10.csv created")

import pandas as pd 
frame = pd.read_csv('perfectmatchdatatop10.csv',index_col=None, header=0)

print ("pandas dataframe for perfectmatch data created")

# for adding Target column and enumerating caller - updates frame

import subprocess

from sklearn.tree import DecisionTreeClassifier, export_graphviz

#Should use this code to add target integer column to my dataframes. http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
        
def encode_rootnode(df, rootnode_column):
    """Add column to df with integers for the root nodes.

    Args
    ----
    df -- pandas DataFrame.
    rootnode_column -- column to map to int, producing
                     new rootnode_num column.

    Returns
    -------
    df_mod -- modified DataFrame.
    rootnode_num -- list of rootnode number names.
    """
    df_mod = df.copy()
    rootnodes = df_mod[rootnode_column].unique()
    map_to_int = {name: n for n, name in enumerate(rootnodes)}
    df_mod["rootnode_num"] = df_mod[rootnode_column].replace(map_to_int)

    return (df_mod, rootnodes)


print ("Amending some columns")

frame2, targets = encode_target(frame, "damsl_act_tag")

print("* targets:", frame2["Target"].unique(), sep="\n")   # 42 targets - 42 DAMSL act tags

frame2['caller'] = frame2['caller'].map({'A': 1, 'B': 2})   # changes caller from A and B to 1 and 2,
                                                            # so that it can be feature in the DT classifier.

frame3, root_nodes = encode_rootnode(frame2, "root_node")

print("* root nodes:", frame3["rootnode_num"].unique(), sep="\n")

features = list((frame3.columns[2], frame3.columns[5], frame3.columns[14], frame3.columns[16]))   # this is just the conv number and sentence length columns
print("* my features:", features, sep="\n")

y = frame3["Target"]
X = frame3[features]
#for min_samples_split, checked the frequency of the top 10 DAs to make sure this was set ok.

# 1) make train and test data split.
# remove act_tag (4) and damsl_act_tag (12) from both. take off Target from test and make labels file/variable.

frame3 = frame3.drop('act_tag', axis=1)
frame3 = frame3.drop('damsl_act_tag', axis=1)

print ("two tag columns dropped")

J = frame3.drop(frame3.columns[[13]], axis=1, inplace=False)
k = frame3.iloc[:,13].to_frame()     # to_frame to keep it as a dataframe, otherwise became a panda series..


#to keep utterances in conversation order (for when adding to the other feature data later):
import numpy as np

def non_shuffling_train_test_split(X, y, test_size=0.25):   #in the end, didn't need this non-shuffled, but did it in case i wanted to investigate order later. not needed to shuffle.
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

J_train, J_test, k_train, k_test = non_shuffling_train_test_split(J, k, test_size=0.25)

#to check:
print("J_train shape:", J_train.shape, "J_test shape:", J_test.shape, "k_train shape:", k_train.shape, "k_test shape:", k_test.shape)

# 2) clean the sentences data:

from KaggleWord2VecUtility import KaggleWord2VecUtility       # this is a file .py saved in the Masters Thesis folder

# Get the number of utterances based on the dataframe column size
num_utterances = J_train["text"].shape[0]

#make empty train data list for the "clean" utterances:
print ("Cleaning and parsing the training set utterances...\n")
clean_train_utterances = []

# Loop over each utterance; create an index i that goes from 0 to the length
# of the utterances list 
for i in range(0, num_utterances):           # changed xrange to range
    # Call our function for each one, and add the result to the list of clean utterances
    clean_train_utterances.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(J_train["text"][i], remove_stopwords=True)))


# 3a) Create Bag-of-Words using CountVectorizer:

from sklearn.feature_extraction.text import CountVectorizer

print ("Creating the bag of words with CountVectorizer...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
c_vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 500)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of          
# strings.
c_train_data_features = c_vectorizer.fit_transform(clean_train_utterances)

# Numpy arrays are easy to work with, so convert the result to an 
# array
c_train_data_features = c_train_data_features.toarray()

#b = J_train['caller'].as_matrix()
#c = J_train['sent_length'].as_matrix()
d = J_train['rootnode_num'].as_matrix()
c_train_data_featurescopy = c_train_data_features.copy()

other_features_train = d[:, None]   # change this variable to test individual variables

c_train_data_features_all = np.hstack((other_features_train,c_train_data_featurescopy))

# 3b) Create Bag-of-Words using TfidfVectorizer:

from sklearn.feature_extraction.text import TfidfVectorizer

print ("Creating the bag of words with TfidfVectorizer...\n")

# Initialize the "TfidfVectorizer" object, which is scikit-learn's
# bag of words tool.  
ti_vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 500) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of          
# strings.
ti_train_data_features = ti_vectorizer.fit_transform(clean_train_utterances)

# Numpy arrays are easy to work with, so convert the result to an 
# array
ti_train_data_features = ti_train_data_features.toarray()

#Now add the 4 non-bag-of-words features from earlier to the bag-of-words features

ti_train_data_featurescopy = ti_train_data_features.copy()

ti_train_data_features_all = np.hstack((other_features_train,ti_train_data_featurescopy))
                             
# 4a) DT Classifier:

# ******* Train 2 decision trees using the 2 bag of words
#
print ("Training the decision trees (this may take a while)...")


# Initialize a Decision Tree classifier
dtree = DecisionTreeClassifier(min_samples_split=20, random_state=99)

# Fit the decision tree to the count and ti-idf training sets, using the 2 bag of words as
# features and the Target labels as the response variable
c_dtree = dtree.fit( c_train_data_features_all, k_train["Target"] )
ti_dtree = dtree.fit( ti_train_data_features_all, k_train["Target"] )

# 4b) RF Classifier:

from sklearn.ensemble import RandomForestClassifier

# ******* Train a random forest using the bag of words
#
print ("Training the random forests (this may take a while)...")


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training sets, using the 2 bag of words as
# features and the Target labels as the response variable
c_forest = forest.fit( c_train_data_features_all, k_train["Target"] )
ti_forest = forest.fit( ti_train_data_features_all, k_train["Target"] ) 

# 5) Test data:

# Get the number of utterances based on the dataframe column size
num_utterances2 = J_test["text"].shape[0]

#make empty test data list for the "clean" utterances:
print ("Cleaning and parsing the test set utterances...\n")
clean_test_utterances = []

# Loop over each utterance; create an index i that goes from 0 to the length
# of the utterances list 
for i in range(0, num_utterances2):           # changed xrange to range
    # Call our function for each one, and add the result to the list of clean utterances
    clean_test_utterances.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(J_test["text"][i+c_train_data_features_all.shape[0]], remove_stopwords=True)))   #doesn't matter here if it's c_ or ti_

# Get 2 bag of words for the test set, and convert to a numpy array for each
c_test_data_features = c_vectorizer.transform(clean_test_utterances)
c_test_data_features = c_test_data_features.toarray()

ti_test_data_features = ti_vectorizer.transform(clean_test_utterances)
ti_test_data_features = ti_test_data_features.toarray()

# three non-BoWs features, stacked together:

#f = J_test['caller'].as_matrix()
#g = J_test['sent_length'].as_matrix()
h = J_test['rootnode_num'].as_matrix()
#other_features_test = np.column_stack((f,g,h))

other_features_test = h[:, None]   # change this variable to test individual variables

# bag of words array (CountVectorizer):
c_test_data_featurescopy = c_test_data_features.copy()
ti_test_data_featurescopy = ti_test_data_features.copy()

# the 4 features plus BoWs features stacked together in one feature array for each of c and ti:
c_test_data_features_all = np.hstack((other_features_test,c_test_data_featurescopy))
ti_test_data_features_all = np.hstack((other_features_test,ti_test_data_featurescopy))

# COUNT: accuracy and other analysis of BoW decision trees
c_pred_cls2 = c_dtree.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)
c_targetlist2 = k_test['Target'].values.tolist()

# TF-IDF: accuracy and other analysis of BoW decision trees
ti_pred_cls2 = ti_dtree.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# COUNT: accuracy and other analysis of BoW random forest
c_pred_cls3 = c_forest.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# TF-IDF: accuracy and other analysis of BoW random forest
ti_pred_cls3 = ti_forest.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# 4ci) a Neural Network: Multi-Layer Perceptron (MLP) Classifier:

# ******* Train an MLP classifier using the bag of words
#
print ("Training the MLP (this may take a while)...")

from sklearn.neural_network import MLPClassifier

# Initialize an MLP classifier
MLP = MLPClassifier(random_state=99)   #same random state as the other classifiers..

# Fit the MLP to the training set, using the bag of words as
# features and the Target labels as the response variable
c_MLP = MLP.fit( c_train_data_features_all, k_train["Target"] )
ti_MLP = MLP.fit( ti_train_data_features_all, k_train["Target"] )

# COUNT: accuracy and other analysis of BoW MLP
c_pred_cls4 = c_MLP.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# TF-IDF: accuracy and other analysis of BoW MLP
ti_pred_cls4 = ti_MLP.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# 4d) Majority Classifier: this is to get a baseline. Predictions are just all the most common label in the training data.

# ******* Train a Majority classifier using the bag of words
#
print ("Training the Majority (this may take a while)...")

from sklearn.dummy import DummyClassifier

# Initialize an Majority classifier
MajFreq = DummyClassifier(strategy="most_frequent", random_state=99)

# Fit the Majority to the training set, using the bag of words as
# features and the Target labels as the response variable
c_MajFreq = MajFreq.fit( c_train_data_features_all, k_train["Target"] )
ti_MajFreq = MajFreq.fit( ti_train_data_features_all, k_train["Target"] )
"""
"""

#strategy=most_frequent   USE THIS ONE, IT'S THE ONE USED IN THE EXAMPLE AND MAKES SENSE AS A BASELINE

# COUNT: accuracy and other analysis of BoW Majority/dummy classifier
c_pred_cls7 = c_MajFreq.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# TF-IDF: accuracy and other analysis of BoW Majority/dummy classifier
ti_pred_cls7 = ti_MajFreq.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# Naive Bayes - Multinomial:

#https://stackoverflow.com/questions/10098533/implementing-bag-of-words-naive-bayes-classifier-in-nltk

# 4f) Naive Bayes Classifier: Multinomial

from sklearn.naive_bayes import MultinomialNB

# ******* Train a NB classifier using the bag of words
#
print ("Training the NB classifiers (this may take a while)...")


# Initialize a NB classifier
NB = MultinomialNB()

# Fit the NB classifier to the training set, using the bag of words as
# features and the Target labels as the response variable
c_NB = NB.fit(c_train_data_features_all, k_train["Target"])
ti_NB = NB.fit(ti_train_data_features_all, k_train["Target"])

# COUNT: accuracy and other analysis of BoW NB
c_pred_cls9 = NB.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)
# TF-IDF: accuracy and other analysis of BoW NB

ti_pred_cls9 = NB.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# 4e) Support Vector Machine (SVM) Classifier:

from sklearn.svm import LinearSVC

# ******* Train a SVM using the bag of words
#
print ("Training the SVM (this may take a while)...")


# Initialize a SVM
SVM = LinearSVC(random_state=99)

# Fit the SVM to the training set, using the bag of words as
# features and the Target labels as the response variable
c_SVM = SVM.fit(c_train_data_features_all, k_train["Target"])
ti_SVM = SVM.fit(ti_train_data_features_all, k_train["Target"])

# COUNT: accuracy and other analysis of BoW SVM
c_pred_cls8 = c_SVM.predict(c_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)

# TF-IDF: accuracy and other analysis of BoW SVM
ti_pred_cls8 = ti_SVM.predict(ti_test_data_features_all)   #e.g. for all utterance (as i haven't indexed), predicts which class (DA)


c_targetlist2train = np.asarray(k_train['Target'].values.tolist())
c_targetlist2test = np.asarray(c_targetlist2)
true_labels = np.hstack((c_targetlist2train, c_targetlist2test))


print("* damsl_act_tag counts - top 10:", frame["damsl_act_tag"].value_counts().nlargest(10), sep="\n")
print("* root_node counts - top 12:", frame["root_node"].value_counts().nlargest(12), sep="\n")


#Cross-validation 10-fold and 5-fold:
from sklearn.model_selection import cross_val_score

c_alldata = np.vstack((c_train_data_features_all, c_test_data_features_all))
ti_alldata = np.vstack((ti_train_data_features_all, ti_test_data_features_all))

#Majority - most_freq
c_MajFreq_cv5 = cross_val_score(c_MajFreq,c_alldata,true_labels,cv=5)
print ("MajFreq countvec - 5-fold CV:", c_MajFreq_cv5)
c_MajFreq_cv10 = cross_val_score(c_MajFreq,c_alldata,true_labels,cv=10)
print ("MajFreq countvec - 10-fold CV:", c_MajFreq_cv10)
ti_MajFreq_cv5 = cross_val_score(ti_MajFreq,ti_alldata,true_labels,cv=5)
print ("MajFreq tfidfvec - 5-fold CV:", ti_MajFreq_cv5)
ti_MajFreq_cv10 = cross_val_score(ti_MajFreq,ti_alldata,true_labels,cv=10)
print ("MajFreq tfidfvec - 10-fold CV:", ti_MajFreq_cv10)

#Decision Tree
c_dtree_cv5 = cross_val_score(c_dtree,c_alldata,true_labels,cv=5)
print ("dtree countvec - 5-fold CV:", c_dtree_cv5)
c_dtree_cv10 = cross_val_score(c_dtree,c_alldata,true_labels,cv=10)
print ("dtree countvec - 10-fold CV:", c_dtree_cv10)
ti_dtree_cv5 = cross_val_score(ti_dtree,ti_alldata,true_labels,cv=5)
print ("dtree tfidfvec - 5-fold CV:", ti_dtree_cv5)
ti_dtree_cv10 = cross_val_score(ti_dtree,ti_alldata,true_labels,cv=10)
print ("dtree tfidfvec - 10-fold CV:", ti_dtree_cv10)

#Random Forest
c_forest_cv5 = cross_val_score(c_forest,c_alldata,true_labels,cv=5)
print ("forest countvec - 5-fold CV:", c_forest_cv5)
c_forest_cv10 = cross_val_score(c_forest,c_alldata,true_labels,cv=10)
print ("forest countvec - 10-fold CV:", c_forest_cv10)
ti_forest_cv5 = cross_val_score(ti_forest,ti_alldata,true_labels,cv=5)
print ("forest tfidfvec - 5-fold CV:", ti_forest_cv5)
ti_forest_cv10 = cross_val_score(ti_forest,ti_alldata,true_labels,cv=10)
print ("forest tfidfvec - 10-fold CV:", ti_forest_cv10)

#MLP
c_MLP_cv5 = cross_val_score(c_MLP,c_alldata,true_labels,cv=5)
print ("MLP countvec - 5-fold CV:", c_MLP_cv5)
c_MLP_cv10 = cross_val_score(c_MLP,c_alldata,true_labels,cv=10)
print ("MLP countvec - 10-fold CV:", c_MLP_cv10)
ti_MLP_cv5 = cross_val_score(ti_MLP,ti_alldata,true_labels,cv=5)
print ("MLP tfidfvec - 5-fold CV:", ti_MLP_cv5)
ti_MLP_cv10 = cross_val_score(ti_MLP,ti_alldata,true_labels,cv=10)
print ("MLP tfidfvec - 10-fold CV:", ti_MLP_cv10)

#Naive Bayes
c_NB_cv5 = cross_val_score(c_NB,c_alldata,true_labels,cv=5)
print ("NB countvec - 5-fold CV:", c_NB_cv5)
c_NB_cv10 = cross_val_score(c_NB,c_alldata,true_labels,cv=10)
print ("NB countvec - 10-fold CV:", c_NB_cv10)
ti_NB_cv5 = cross_val_score(ti_NB,ti_alldata,true_labels,cv=5)
print ("NB tfidfvec - 5-fold CV:", ti_NB_cv5)
ti_NB_cv10 = cross_val_score(ti_NB,ti_alldata,true_labels,cv=10)
print ("NB tfidfvec - 10-fold CV:", ti_NB_cv10)

#Linear SVM
c_SVM_cv5 = cross_val_score(c_SVM,c_alldata,true_labels,cv=5)
print ("SVM countvec - 5-fold CV:", c_SVM_cv5)
c_SVM_cv10 = cross_val_score(c_SVM,c_alldata,true_labels,cv=10)
print ("SVM countvec - 10-fold CV:", c_SVM_cv10)
ti_SVM_cv5 = cross_val_score(ti_SVM,ti_alldata,true_labels,cv=5)
print ("SVM tfidfvec - 5-fold CV:", ti_SVM_cv5)
ti_SVM_cv10 = cross_val_score(ti_SVM,ti_alldata,true_labels,cv=10)
print ("SVM tfidfvec - 10-fold CV:", ti_SVM_cv10)

print("Accuracy Top10 - c_MajFreq_cv5: %0.2f (+/- %0.2f)" % (c_MajFreq_cv5.mean(), c_MajFreq_cv5.std() * 2))
print("Accuracy Top10 - c_MajFreq_cv10: %0.2f (+/- %0.2f)" % (c_MajFreq_cv10.mean(), c_MajFreq_cv10.std() * 2))
print("Accuracy Top10 - ti_MajFreq_cv5: %0.2f (+/- %0.2f)" % (ti_MajFreq_cv5.mean(), ti_MajFreq_cv5.std() * 2))
print("Accuracy Top10 - ti_MajFreq_cv10: %0.2f (+/- %0.2f)" % (ti_MajFreq_cv10.mean(), ti_MajFreq_cv10.std() * 2))

print("Accuracy Top10 - c_dtree_cv5: %0.2f (+/- %0.2f)" % (c_dtree_cv5.mean(), c_dtree_cv5.std() * 2))
print("Accuracy Top10 - c_dtree_cv10: %0.2f (+/- %0.2f)" % (c_dtree_cv10.mean(), c_dtree_cv10.std() * 2))
print("Accuracy Top10 - ti_dtree_cv5: %0.2f (+/- %0.2f)" % (ti_dtree_cv5.mean(), ti_dtree_cv5.std() * 2))
print("Accuracy Top10 - ti_dtree_cv10: %0.2f (+/- %0.2f)" % (ti_dtree_cv10.mean(), ti_dtree_cv10.std() * 2))

print("Accuracy Top10 - c_forest_cv5: %0.2f (+/- %0.2f)" % (c_forest_cv5.mean(), c_forest_cv5.std() * 2))
print("Accuracy Top10 - c_forest_cv10: %0.2f (+/- %0.2f)" % (c_forest_cv10.mean(), c_forest_cv10.std() * 2))
print("Accuracy Top10 - ti_forest_cv5: %0.2f (+/- %0.2f)" % (ti_forest_cv5.mean(), ti_forest_cv5.std() * 2))
print("Accuracy Top10 - ti_forest_cv10: %0.2f (+/- %0.2f)" % (ti_forest_cv10.mean(), ti_forest_cv10.std() * 2))

print("Accuracy Top10 - c_MLP_cv5: %0.2f (+/- %0.2f)" % (c_MLP_cv5.mean(), c_MLP_cv5.std() * 2))
print("Accuracy Top10 - c_MLP_cv10: %0.2f (+/- %0.2f)" % (c_MLP_cv10.mean(), c_MLP_cv10.std() * 2))
print("Accuracy Top10 - ti_MLP_cv5: %0.2f (+/- %0.2f)" % (ti_MLP_cv5.mean(), ti_MLP_cv5.std() * 2))
print("Accuracy Top10 - ti_MLP_cv10: %0.2f (+/- %0.2f)" % (ti_MLP_cv10.mean(), ti_MLP_cv10.std() * 2))

print("Accuracy Top10 - c_NB_cv5: %0.2f (+/- %0.2f)" % (c_NB_cv5.mean(), c_NB_cv5.std() * 2))
print("Accuracy Top10 - c_NB_cv10: %0.2f (+/- %0.2f)" % (c_NB_cv10.mean(), c_NB_cv10.std() * 2))
print("Accuracy Top10 - ti_NB_cv5: %0.2f (+/- %0.2f)" % (ti_NB_cv5.mean(), ti_NB_cv5.std() * 2))
print("Accuracy Top10 - ti_NB_cv10: %0.2f (+/- %0.2f)" % (ti_NB_cv10.mean(), ti_NB_cv10.std() * 2))

print("Accuracy Top10 - c_SVM_cv5: %0.2f (+/- %0.2f)" % (c_SVM_cv5.mean(), c_SVM_cv5.std() * 2))
print("Accuracy Top10 - c_SVM_cv10: %0.2f (+/- %0.2f)" % (c_SVM_cv10.mean(), c_SVM_cv10.std() * 2))
print("Accuracy Top10 - ti_SVM_cv5: %0.2f (+/- %0.2f)" % (ti_SVM_cv5.mean(), ti_SVM_cv5.std() * 2))
print("Accuracy Top10 - ti_SVM_cv10: %0.2f (+/- %0.2f)" % (ti_SVM_cv10.mean(), ti_SVM_cv10.std() * 2))

#metrics scores
from sklearn.metrics import precision_recall_fscore_support

# weighted accuracies for the decision tree:
cprecisionDTw, crecallDTw, cfscoreDTw, csupportDTw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls2,average='weighted')
tprecisionDTw, trecallDTw, tfscoreDTw, tsupportDTw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls2,average='weighted')

# unweighted accuracies for the decision tree:
cprecisionDTu, crecallDTu, cfscoreDTu, csupportDTu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls2)
tprecisionDTu, trecallDTu, tfscoreDTu, tsupportDTu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls2)

# weighted accuracies for the random forest:
cprecisionRFw, crecallRFw, cfscoreRFw, csupportRFw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls3,average='weighted')
tprecisionRFw, trecallRFw, tfscoreRFw, tsupportRFw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls3,average='weighted')

# unweighted accuracies for the random forest:
cprecisionRFu, crecallRFu, cfscoreRFu, csupportRFu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls3)
tprecisionRFu, trecallRFu, tfscoreRFu, tsupportRFu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls3)

# weighted accuracies for the MLP:
cprecisionMLPw, crecallMLPw, cfscoreMLPw, csupportMLPw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls4,average='weighted')
tprecisionMLPw, trecallMLPw, tfscoreMLPw, tsupportMLPw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls4,average='weighted')

# unweighted accuracies for the MLP:
cprecisionMLPu, crecallMLPu, cfscoreMLPu, csupportMLPu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls4)
tprecisionMLPu, trecallMLPu, tfscoreMLPu, tsupportMLPu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls4)

# weighted accuracies for the NB:
cprecisionNBw, crecallNBw, cfscoreNBw, csupportNBw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls9,average='weighted')
tprecisionNBw, trecallNBw, tfscoreNBw, tsupportNBw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls9,average='weighted')

# unweighted accuracies for the NB:
cprecisionNBu, crecallNBu, cfscoreNBu, csupportNBu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls9)
tprecisionNBu, trecallNBu, tfscoreNBu, tsupportNBu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls9)

# weighted accuracies for the Majority Most Frequent classifier:
cprecisionMFw, crecallMFw, cfscoreMFw, csupportMFw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls7,average='weighted')
tprecisionMFw, trecallMFw, tfscoreMFw, tsupportMFw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls7,average='weighted')

# unweighted accuracies for the Majority Most Frequent classifier:
cprecisionMFu, crecallMFu, cfscoreMFu, csupportMFu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls7)
tprecisionMFu, trecallMFu, tfscoreMFu, tsupportMFu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls7)

# weighted accuracies for the SVM:
cprecisionSVMw, crecallSVMw, cfscoreSVMw, csupportSVMw = precision_recall_fscore_support(c_targetlist2test,c_pred_cls8,average='weighted')
tprecisionSVMw, trecallSVMw, tfscoreSVMw, tsupportSVMw = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls8,average='weighted')

# unweighted accuracies for the SVM:
cprecisionSVMu, crecallSVMu, cfscoreSVMu, csupportSVMu = precision_recall_fscore_support(c_targetlist2test,c_pred_cls8)
tprecisionSVMu, trecallSVMu, tfscoreSVMu, tsupportSVMu = precision_recall_fscore_support(c_targetlist2test,ti_pred_cls8)


print("Count DT weighted accuracies:",cprecisionDTw, crecallDTw, cfscoreDTw, csupportDTw)
print("TF-IDF DT weighted accuracies:",tprecisionDTw, trecallDTw, tfscoreDTw, tsupportDTw)
print("Count RF weighted accuracies:",cprecisionRFw, crecallRFw, cfscoreRFw, csupportRFw)
print("TF-IDF RF weighted accuracies:",tprecisionRFw, trecallRFw, tfscoreRFw, tsupportRFw)
print("Count MLP weighted accuracies:",cprecisionMLPw, crecallMLPw, cfscoreMLPw, csupportMLPw)
print("TF-IDF MLP weighted accuracies:",tprecisionMLPw, trecallMLPw, tfscoreMLPw, tsupportMLPw)
print("Count NB weighted accuracies:",cprecisionNBw, crecallNBw, cfscoreNBw, csupportNBw)
print("TF-IDF NB weighted accuracies:",tprecisionNBw, trecallNBw, tfscoreNBw, tsupportNBw)
print("Count MF weighted accuracies:",cprecisionMFw, crecallMFw, cfscoreMFw, csupportMFw)
print("TF-IDF MF weighted accuracies:",tprecisionMFw, trecallMFw, tfscoreMFw, tsupportMFw)
print("Count SVM weighted accuracies:",cprecisionSVMw, crecallSVMw, cfscoreSVMw, csupportSVMw)
print("TF-IDF SVM weighted accuracies:",tprecisionSVMw, trecallSVMw, tfscoreSVMw, tsupportSVMw)


print("Count DT unweighted accuracies:",cprecisionDTu, crecallDTu, cfscoreDTu, csupportDTu)
print("TF-IDF DT unweighted accuracies:",tprecisionDTu, trecallDTu, tfscoreDTu, tsupportDTu)
print("Count RF unweighted accuracies:",cprecisionRFu, crecallRFu, cfscoreRFu, csupportRFu)
print("TF-IDF RF unweighted accuracies:",tprecisionRFu, trecallRFu, tfscoreRFu, tsupportRFu)
print("Count MLP unweighted accuracies:",cprecisionMLPu, crecallMLPu, cfscoreMLPu, csupportMLPu)
print("TF-IDF MLP unweighted accuracies:",tprecisionMLPu,trecallMLPu, tfscoreMLPu, tsupportMLPu)
print("Count NB unweighted accuracies:",cprecisionNBu, crecallNBu, cfscoreNBu, csupportNBu)
print("TF-IDF NB unweighted accuracies:",tprecisionNBu, trecallNBu, tfscoreNBu, tsupportNBu)
print("Count MF unweighted accuracies:",cprecisionMFu, crecallMFu, cfscoreMFu, csupportMFu)
print("TF-IDF MF unweighted accuracies:",tprecisionMFu, trecallMFu, tfscoreMFu, tsupportMFu)
print("Count SVM unweighted accuracies:",cprecisionSVMu, crecallSVMu, cfscoreSVMu, csupportSVMu)
print("TF-IDF SVM unweighted accuracies:",tprecisionSVMu, trecallSVMu, tfscoreSVMu, tsupportSVMu)

# Confusion matrix

from sklearn.metrics import confusion_matrix

c_DTconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls2)
c_RFconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls3)
c_MLPconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls4)
c_NBconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls9)
c_MFconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls7)
c_SVMconfmatrix = confusion_matrix(c_targetlist2test,c_pred_cls8)

ti_DTconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls2)
ti_RFconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls3)
ti_MLPconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls4)
ti_NBconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls9)
ti_MFconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls7)
ti_SVMconfmatrix = confusion_matrix(c_targetlist2test,ti_pred_cls8)


#from plot_confusion_matrix import plot_confusion_matrix

#need more code for this, e.g. plt.figure() and plt.show()
#plot_confusion_matrix(DTconfmatrix,classes=targetlist2,title="Decision Tree Confusion Matrix")