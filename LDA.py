import numpy as np
from util import *


class LatentDirichletAllocation:
    """

    K: number of topics (int)
    W: size of vocabulary (int)
    D: number of documents (int)
    corpus: corpus (list of list)
    vocabulary: different words (dictionary)
    mapping: map from words to a unique number (dictionary)
    n_dk: number of words assigned to topic k in document d (np array (D, K))
    m_kw: number of word w assigned to topic k in document (np array (K, W))
    n_d: number of words in d (np array (D, 1))
    m_k: number of words assigned to topic k in all documents (np array (K, 1))
    assignment: assignment of topics for each word


    """

    def __init__(self, dir, n_topics, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.K = n_topics
        self.corpus = read_doc_lda(dir)
        self.D = len(self.corpus)
        self.vocabulary = initial_dictionary_lda(self.corpus)
        self.mapping = generate_mapping(self.vocabulary)
        self.W = len(self.vocabulary)
        self.n_dk, self.m_kw, self.assignment = initial_assignment_lda(self.corpus, self.K, self.W, self.D, self.mapping)
        self.n_d = np.sum(self.n_dk, axis=1)
        self.m_k = np.sum(self.m_kw, axis=1)


    def gibbs_sampler(self, n_burn_in=10):
        for i in range(n_burn_in):
            for j in range(self.D):
                for k in range(len(self.corpus[j])):
                    current_topic = self.assignment[j][k]
                    current_word = self.corpus[j][k]
                    current_word_mapping = self.mapping[current_word]
                    self.n_d[j] -= 1
                    self.n_dk[j, current_topic] -= 1
                    self.m_k[current_topic] -= 1
                    self.m_kw[current_topic, current_word_mapping] -= 1
                    prob = conditional_prob(self.n_dk, self.n_d, self.m_kw, self.m_k, j, current_word_mapping,
                                            self.alpha, self.beta)
                    new_topic = sample_topic(prob)
                    self.assignment[j][k] = new_topic
                    self.n_d[j] += 1
                    self.n_dk[j, new_topic] += 1
                    self.m_k[new_topic] += 1
                    self.m_kw[new_topic, current_word_mapping] += 1
            print('iteration ' + str(i))


    def get_model(self):
        model = {}
        for word in self.vocabulary:
            model[word] =[]
            model[word].append(np.matrix(self.m_kw[:, self.mapping[word]]).T.tolist())
        f = open('LDA_model_mini.txt', 'wt')
        for word in sorted(model.keys()):
            f.write(word)
            for i in range(len(model[word][0])):
                f.write(' '+str(model[word][0][i][0]))
            f.write('\n')
        f.close()
        return model

    def get_result(self):
        return self.assignment


# lda = LatentDirichletAllocation('corpus_1.txt', 4, 10, 0.01)
# lda.gibbs_sampler(150)
# lda.get_model()


