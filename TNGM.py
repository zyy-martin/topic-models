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
                for k in range(len(self.corpus[j])):
                    for l in range(len(self.corpus[j][k])):
                        current_topic = self.assignment[j][k][l]
                        current_word = self.corpus[j][k][l]
                        current_word_mapping = self.mapping[current_word]

                        # if the first word in a sentence
                        if l == 0:
                            #
                            #
                            # ======== update z =======
                            #
                            #
                            self.q_dk[j, current_topic] -= 1
                            self.n_kw[current_topic, current_word_mapping] -= 1
                            self.n_k[current_topic] -= 1
                            prob_z = conditional_prob_z_x0(self.alpha, self.beta,self.q_dk,self.n_kw,self.n_k,j,current_word_mapping)

                            new_topic = sample_topic(prob_z)
                            self.assignment[j][k][l] = new_topic
                            self.q_dk[j, new_topic] += 1
                            self.n_kw[new_topic, current_word_mapping] += 1
                            self.n_k[new_topic] += 1

                            if l < len(self.corpus[j][k])-1:
                                next_x = self.x[j][k][l]
                                self.p_kwx[current_topic, current_word_mapping, next_x] -= 1
                                self.p_kwx[new_topic, current_word_mapping, next_x] += 1

                            # finish updating
                            continue

                        # if not the first word

                        current_x = self.x[j][k][l - 1]

                        # if current x = 0
                        if current_x == 0:
                            # get previous word
                            prev_word = self.corpus[j][k][l-1]
                            prev_word_mapping = self.mapping[prev_word]

                            # get previous topic
                            prev_topic = self.assignment[j][k][l-1]


                            #
                            #
                            # ======== update z =======
                            #
                            #

                            self.q_dk[j, current_topic] -= 1
                            self.n_kw[current_topic, current_word_mapping] -= 1
                            self.n_k[current_topic] -= 1


                            prob_z = conditional_prob_z_x0(self.alpha, self.beta, self.q_dk, self.n_kw, self.n_k, j,
                                                           current_word_mapping)

                            new_topic = sample_topic(prob_z)

                            self.assignment[j][k][l] = new_topic

                            # update counts
                            self.q_dk[j, new_topic] += 1
                            self.n_kw[new_topic, current_word_mapping] += 1
                            self.n_k[new_topic] += 1

                            if l < len(self.corpus[j][k])-1:
                                next_x = self.x[j][k][l]
                                self.p_kwx[current_topic, current_word_mapping, next_x] -= 1
                                self.p_kwx[new_topic, current_word_mapping, next_x] += 1


                            #
                            #
                            # ======== update x =======
                            #
                            #

                            self.p_kwx[prev_topic, prev_word_mapping, 0] -= 1



                            prob_x = conditional_prob_x_x0(self.gamma, self.beta, self.p_kwx, self.n_kw, self.n_k, current_word_mapping, prev_word_mapping, new_topic, prev_topic)
                            new_x = sample_topic(prob_x)

                            self.x[j][k][l-1] = new_x

                            # update counts z
                            # self.q_dk[j, new_topic] += 1
                            # self.n_kw[new_topic, current_word_mapping] += 1
                            # self.n_k[new_topic] += 1

                            # update counts x
                            self.p_kwx[prev_topic, prev_word_mapping, new_x] += 1



                        else:
                            # get previous word
                            prev_word = self.corpus[j][k][l - 1]
                            prev_word_mapping = self.mapping[prev_word]

                            # get previous topic
                            prev_topic = self.assignment[j][k][l - 1]

                            #
                            #
                            # ======== update z =======
                            #
                            #

                            self.q_dk[j, current_topic] -= 1
                            self.m_kwv[current_topic, prev_word_mapping, current_word_mapping] -= 1
                            self.m_kw[current_topic, prev_word_mapping] -= 1
                            prob_z = conditional_prob_z_x1(self.alpha, self.sigma, self.q_dk,self.m_kwv,self.m_kw, j, current_word_mapping, prev_word_mapping)
                            new_topic = sample_topic(prob_z)

                            self.assignment[j][k][l] = new_topic

                            # update counts
                            self.q_dk[j, new_topic] += 1
                            self.m_kwv[new_topic, prev_word_mapping, current_word_mapping] += 1
                            self.m_kw[new_topic, prev_word_mapping] += 1

                            if l < len(self.corpus[j][k])-1:
                                next_x = self.x[j][k][l]
                                self.p_kwx[current_topic, current_word_mapping, next_x] -= 1
                                self.p_kwx[new_topic, current_word_mapping, next_x] += 1

                            #
                            #
                            # ======== update x =======
                            #
                            #

                            self.p_kwx[prev_topic, prev_word_mapping, 1] -= 1


                            prob_x = conditional_prob_x_x1(self.gamma, self.sigma, self.p_kwx, self.m_kwv, self.m_kw, current_word_mapping, prev_word_mapping, new_topic, prev_topic)

                            new_x = sample_topic(prob_x)
                            self.x[j][k][l - 1] = new_x

                            # update counts z
                            # self.q_dk[j, new_topic] += 1
                            # self.m_kwv[new_topic, prev_word_mapping, current_word_mapping] += 1
                            # self.m_kw[new_topic, prev_word_mapping] += 1

                            # update counts x
                            self.p_kwx[prev_topic, prev_word_mapping, new_x] += 1
            print('iteration ' + str(i))


    # def get_assignment(self):
    #     sequence_dic = {}
    #     for i in range(self.D):
    #         for j in range(len(self.corpus[i])):
    #             sequence = ''
    #             sequence_topic = -1
    #             for k in range(len(self.corpus[i][j])):
    #                 if sequence_topic == -1:
    #                     sequence += ' '+self.corpus[i][j][k]
    #                     sequence_topic = self.assignment[i][j][k]
    #                 else:
    #                     if self.x[i][j][k-1] == 1:
    #                         sequence += ' '+self.corpus[i][j][k]
    #                         sequence_topic = self.assignment[i][j][k]
    #                     else:
    #                         if sequence in sequence_dic:
    #                             sequence_dic[sequence][sequence_topic] += 1
    #                             sequence = ''
    #                             sequence_topic = -1
    #                         else:
    #                             sequence_dic[sequence] = [0] * self.K
    #                             sequence_dic[sequence][sequence_topic] += 1
    #                             sequence = ''
    #                             sequence_topic = -1
    #             if sequence_topic == -1:
    #                 continue
    #             else:
    #                 if sequence in sequence_dic:
    #                     sequence_dic[sequence][sequence_topic] += 1
    #                 else:
    #                     sequence_dic[sequence] = [0] * self.K
    #                     sequence_dic[sequence][sequence_topic] += 1
    #
    #     f = open('assignment.txt', 'wt')
    #     for sequence in sequence_dic:
    #         f.write(sequence+' ')
    #         for count in sequence_dic[sequence]:
    #             f.write(str(count)+' ')
    #         f.write('\n')
    #     f.close()




    def get_sequence(self):
        sequence_dic = {}
        for i in range(self.D):
            for j in range(len(self.corpus[i])):
                sequence = []
                for k in range(len(self.corpus[i][j])):
                    if k == 0:
                        sequence.append(self.assignment[i][j][k])
                    elif self.x[i][j][k-1] == 0:
                        word = list_to_string(sequence)

                        if word in sequence_dic:
                            sequence_dic[word] += 1
                            sequence = [self.assignment[i][j][k]]
                        else:
                            sequence_dic[word] = 1
                            sequence = [self.assignment[i][j][k]]
                    else:
                        sequence.append(self.assignment[i][j][k])
                word = list_to_string(sequence)
                if word == '':
                    continue
                elif word in sequence_dic:
                    sequence_dic[word] += 1
                else:
                    sequence_dic[word] = 1

        print(sequence_dic)
        f = open('sequence_tngm_all.txt', 'wt')
        for sequence in sorted(sequence_dic.keys()):
            # for i in range(sequence_dic[sequence]):
            #     f.write(sequence+' ')
            f.write(sequence+' '+str(sequence_dic[sequence]))
            f.write('\n')
        f.close()
        return sequence_dic

    def check(self):
        total_words = 0
        total_connection = 0
        for i in range(self.D):
            for j in range(len(self.corpus[i])):
                total_words += len(self.corpus[i][j])
                total_connection += sum(self.x[i][j])
        print(total_words, total_connection)




model = TopicalNGramModel('corpus.txt', 5, 10, 0.01, 0.01, 0.01)
model.gibbs_sampler(100)
model.get_sequence()
model.check()


