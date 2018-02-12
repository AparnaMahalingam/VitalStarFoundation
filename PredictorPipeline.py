

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
import statsmodels.api as sm
from sklearn import metrics

import logging

class PredictorPipeline(object):
    def __init__(self, training, testing):

        self.training = training
        self.testing = testing

    def __set_binary_label(self, target_label, stage_name):
        '''
        In a multi-label training set, we can set up a binary classifier by using a target_label, which is one of
        the multi-labels.
        :param labels_vector: original labels for each training point
        :param stage_name: either 'training' or 'testing' stage
        :return: (target_label, positive size, negative size)
        '''
        if stage_name == 'training':
            labels_vector = self.training['intent']
        else:
            labels_vector = self.testing['intent']

        total_labeled_training_points = labels_vector.shape[0]
        logging.warning( 'total_labeled_training_points = ' + str(total_labeled_training_points) )

        number_of_labels = len(labels_vector.value_counts())
        logging.warning( 'number_of_labels =' + str(number_of_labels) )

        logging.warning( 'target_label ="' + str(target_label) + '"' )

        positive_indicator = np.array(labels_vector == target_label)
        positive_training = np.sum(positive_indicator)
        logging.warning( 'positive_training = ' + str(positive_training) )

        negative_training = total_labeled_training_points - positive_training
        logging.warning( 'negative_training = ' + str(negative_training) )

        return positive_indicator, labels_vector

    def __tokenize_utterance(self, utterance):
        tokenized_utterance = []
        for line in utterance:
            # line_elements =

            elements = line.split(" ")
            tokenized_utterance.append(elements)
        return tokenized_utterance

    def __extract_feature_from_sentence_using_w2v(self, sentence, model):
        '''
        For each sentence, generate a feature vector using a pretrained word vector embedding.
        The algorithm uses the average of all valid words in the sentence.
        :param sentence: array of word tokens that represent a sentence
        :param model: a pretrained w2v model
        :return: feature vector (a numpy array) for the sentence
        '''

        dim_of_model = model.vector_size
        len_of_sentence = len(sentence)
        feature_vec_for_the_sentence = [0] * dim_of_model
        for word in sentence:
            try:
                vec = model.wv.word_vec(word)
                feature_vec_for_the_sentence += vec
            except:
                # for word that's not in the dictionary of w2v, we skip and reduce the len of sentence by 1
                len_of_sentence -= 1
                pass

        if len_of_sentence == 0:
            return [np.nan] * dim_of_model
        else:
            return np.array(feature_vec_for_the_sentence) / len_of_sentence

    def build_binary_classifying_regressors_using_w2v_from_corpus(self, target_label, model, corpus, stage):
        '''
        Build up the training set by extracting the NLP features for each training point and then turn them into
        regressors.

        :param target_label: the label to classify
        :param model: the word2vec model to use
        :param corpus: array of sentences, each of which represent a training instance
        :return:
        '''

        #self.training['utterance']
        w2v_model = model
        corpus_training_processed = self.__tokenize_utterance(corpus)
        positive_indicator, labels_vector = self.__set_binary_label( target_label, stage_name=stage)

        alldocs_training = []
        ChatLine = namedtuple('ChatLine', 'words tags label label_ind feature')
        for i, doc in enumerate(corpus_training_processed):
            feature = self.__extract_feature_from_sentence_using_w2v(doc, w2v_model)
            chat = ChatLine(doc, [i], labels_vector[i], positive_indicator[i], feature)
            alldocs_training.append(chat)

        train_targets, train_regressors = zip(
            *[(doc.label_ind, doc.feature) for doc in alldocs_training if np.sum(np.isnan(doc.feature) == 0)])
        train_regressors = sm.add_constant(train_regressors)

        return train_targets, train_regressors

    def roc_curve(self, test_predictions, test_targets, target_label):

        false_positives = np.array([0] * 10, dtype='d')
        true_positives = np.array([0] * 10, dtype='d')
        false_negatives = np.array([0] * 10, dtype='d')
        true_negatives = np.array([0] * 10, dtype='d')

        accuracy = np.array([0] * 10, dtype='d')
        threshold = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        for j, thres in enumerate(threshold):
            for i, doc in enumerate(test_targets):
                if doc and test_predictions[i] >= thres:
                    # print "correct prediction", test_predictions[i], doc.intents, doc.words
                    true_positives[j] += 1
                if not doc and test_predictions[i] >= thres:
                    # print "false positive", test_predictions[i], doc.intents, doc.words
                    false_positives[j] += 1

            for i, doc in enumerate(test_targets):
                if doc and test_predictions[i] < thres:
                    false_negatives[j] += 1
                    # print doc.intent1, test_predictions[i], doc.intents, doc.words
                if not doc and test_predictions[i] < thres:
                    true_negatives[j] += 1


        true_labels = np.sum(test_targets)
        false_labels = len(test_targets) - true_labels
        print true_labels
        print false_labels

        fpr = false_positives / (false_positives + true_negatives)
        fdr = false_positives / (false_positives + true_positives)
        tpr = true_positives / true_labels

        for j, thres in enumerate(threshold):
            corrects = sum((test_predictions >= thres) == test_targets)
            errors = len(test_predictions) - corrects
            error_rate = float(errors) / len(test_predictions)
            accuracy[j] = 1 - error_rate

        roc = [('threshold', threshold),
               ('false_positives', false_positives),
               ('true_positives', true_positives),
               ('false_negatives', false_negatives),
               ('true_negatives', true_negatives),
               ('false_positive_rate', fpr),
               ('false_discovery_rate', fdr),
               ('true_positive_rate', tpr),
               ('precision', 1 - fdr),
               ('recall', tpr),
               ('accuracy', accuracy)
               ]
        df = pd.DataFrame.from_items(roc)

        plt.scatter(1 - fdr, tpr)
        plt.plot(1 - fdr, tpr)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title('precision-recall curve for ' + target_label)
        plt.show()
        print metrics.auc(1 - fdr, tpr)

        plt.scatter(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('roc curve for ' + target_label)
        plt.show()

        print metrics.auc(fpr, tpr)
        return df

