import numpy as np
from util import *


class TopicalNGramModel:
    """

    K: number of topics (int)
    W: size of vocabulary (int)
    D: number of documents (int)
    corpus: corpus (list of list)
    vocabulary: different words (dictionary)
    mapping: map from words to a unique number (dictionary)
    q_dk: number of words assigned to topic k in document d (np array (D, K))
    n_kw: number of word w assigned to topic k in all documents (np array (K, W))
    m_kwv: number of word v assigned to topic k with previous word w in all documents (np array (K, W, W))
    p_kwx: number of x_v = x with v assigned to topic k and previous word w in all documents (np array (K, W, 2))
    n_k: number of words assigned to topic k (np array (W, 1))
    m_kw: number of words assigned to topic k with previous word w (np array (K, W))
    assignment: assignment of topics for each word


    """
    def __init__(self, dir, n_topics, alpha, beta, gamma, sigma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.K = n_topics
        self.corpus = read_doc_tngm(dir)
        self.D = len(self.corpus)
        self.vocabulary = initial_dictionary_tngm(self.corpus)
        self.mapping = generate_mapping(self.vocabulary)
        self.W = len(self.vocabulary)
        self.q_dk, self.n_kw, self.m_kwv, self.p_kwx, self.assignment, self.x \
            = initial_assignment_tngm(self.corpus, self.K,self.W, self.D, self.mapping)
        self.n_k = np.sum(self.n_kw, axis=1)
        self.m_kw = np.sum(self.m_kwv, axis=2)

    def gibbs_sampler(self, n_burn_in=10):
        for i in range(n_burn_in):
            for j in range(self.D):
                for k in range(self.corpus[j]):
                    for l in range(self.corpus[j][k]):
                        current_topic = self.assignment[j][k][l]
                        current_word = self.corpus[j][k][l]
                        current_word_mapping = self.mapping[current_word]
                        current_x = 0
                        if l > 0:
                            current_x = self.x[j][k][l-1]

                        if current_x == 0:
                            # update z
                            self.q_dk[j, current_topic] -= 1
                            self.n_kw[current_topic, current_word_mapping] -= 1
                            self.n_k[current_topic] -= 1










model = TopicalNGramModel('test_doc_tngm.txt', 3, 0.01, 0.01, 0.01, 0.01)
print(model.corpus)


