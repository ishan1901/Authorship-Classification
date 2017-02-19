import re,sys
import os
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random
import nltk
from collections import Counter


##################################### Features ####################################################

#returns the number of sentences in text
def count_sentences(sentences):
    number_sentences = len(sentences);
    return number_sentences;
        
#returns the number of tokens in text
def count_tokens(tokens):
    number_tokens = len(tokens);
    return number_tokens;

#returns average sentence length
def avg_sentence_length(sentences):
    total_length = 0;
    for sentence in sentences:
        total_length = total_length + count_tokens(sentence);
    return (float(total_length))/float(len(sentence));

#returns average token length
def avg_token_length(tokens):
    total_length = 0;
    for token in tokens:
        total_length = total_length + len(token);
    return (float(total_length))/(float(len(tokens)));

#returns number of pronouns, semicolons, Exclamation Marks, periods, question marks, commas
def get_stylometric(text):
    #store in a dictionary
    all_features = []
    count_pronouns = 0;
    count_semicolons = 0;
    count_exclamation = 0;
    count_periods = 0;
    count_question_marks = 0;
    count_commas = 0;
    for line in text:
        line = line.lower();
        count_pronouns = count_pronouns + len(re.findall('he|him|his|himself|she|her|hers|herself|it|its|itself',line));
        count_semicolons = count_semicolons + len(re.findall(';',line));
        count_exclamation = count_exclamation + len(re.findall('!',line));  
        count_periods = count_periods + len(re.findall('\.',line));
        count_question_marks = count_question_marks + len(re.findall('\?',line));
        count_commas = count_commas + len(re.findall(',',line));
    all_features.append(count_pronouns);
    all_features.append(count_semicolons);
    all_features.append(count_exclamation);
    all_features.append(count_periods);
    all_features.append(count_question_marks);
    all_features.append(count_commas);
    return all_features
"""
#returns number of pronouns
def count_pronouns(text):
    count_pronouns = 0;
    for line in text:
        line = line.lower();
        count_pronouns = count_pronouns + len(re.findall('he|him|his|himself|she|her|hers|herself|it|its|itself',line));

    return count_pronouns;  

#returns number of semicolons
def count_semicolons(text):
    count_semicolons = 0;
    for line in text:
        count_semicolons = count_semicolons + len(re.findall(';',line));
    return count_semicolons;

#returns number of Exclamation Marks
def count_exclamationMarks(text):
    count_exclamation = 0;
    for line in text:
        count_exclamation = count_exclamation + len(re.findall('!',line));
    return count_exclamation;

#returns number of periods
def count_periods(text):
    count_periods = 0;
    for line in text:
        count_periods = count_periods + len(re.findall('\.',line));
    return count_periods;

#returns number of question marks
def count_question_marks(text):
    count_question_marks = 0;
    for line in text:
        count_question_marks = count_question_marks + len(re.findall('\?',line));
    return count_question_marks;
    
#returns number of commas
def count_commas(text):
    count_commas = 0;
    for line in text:
        count_commas = count_commas + len(re.findall(',',line));
    return count_commas;               
"""
def pos_tags(text):
    tokens = nltk.word_tokenize(text)
    return (" ".join( [ tag for (word, tag) in nltk.pos_tag( tokens ) ] ));         

 #returns an array with feature values
def feature_array_function(text):
    tokens = nltk.word_tokenize(text);
    sentences = nltk.sent_tokenize(text);
    get_other_features = [];
    get_other_features.append(get_stylometric(text));
    feat_array=[];
    feat_array.append(count_sentences(sentences));       #1  number of sentences
    feat_array.append(count_tokens(tokens));             #2  number of tokens
    feat_array.append(avg_sentence_length(sentences));   #3  average sentence length
    feat_array.append(avg_token_length(tokens));         #4  average token length
    feat_array.append(get_other_features[0][0]);         #5  frequency of pronouns
    feat_array.append(get_other_features[0][1]);         #6  frequency of semicolons
    feat_array.append(get_other_features[0][2]);         #7  frequency of exclamation marks
    feat_array.append(get_other_features[0][3]);         #8  frequency of periods
    feat_array.append(get_other_features[0][4]);         #9  frequency of question marks
    feat_array.append(get_other_features[0][5]);         #10 frequency of commas

    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));

    
    return feat_array;

###################################################################
#####################bag of words##################################
###################################################################

# def bag_of_words_train(trainData):
# #bag of words features
#     count_vect = CountVectorizer();
#     X_train_counts = count_vect.fit_transform(trainingBunch.data);
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts);
# print (X_train_tfidf.shape);