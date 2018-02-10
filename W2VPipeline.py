import gensim
import logging

class W2VPipeline(object):

    def __init__(self, training):
        self.corpus = training

    def train(self):

        utterance = self.corpus['utterance']
        tokenized_utterance = []
        for line in utterance:
            # line_elements =

            elements = line.split(" ")
            tokenized_utterance.append(elements)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = gensim.models.Word2Vec(tokenized_utterance, min_count=5)
        return model