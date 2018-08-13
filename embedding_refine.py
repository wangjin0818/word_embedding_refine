from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim
import codecs
import argparse

import numpy as np
import pandas as pd

def update_dict(target_dict, key_a, key_b, val): 
    if key_a in target_dict:
        target_dict[key_a].update({key_b: val})
    else:
        target_dict.update({key_a:{key_b: val}})

def write_vector(file_name, w2v_model, vertex_matrix, anew_dict):
    with codecs.open(file_name, 'w', 'utf8') as my_file:
        # refine result
        for i in range(len(anew_dict.keys())):
            word = anew_dict.keys()[i]
            vec = vertex_matrix[i]
            my_file.write('%s %s\n' % (word, ' '.join('%f' % val for val in vec)))

        for word in w2v_model.vocab:
            if word not in anew_dict.keys():
                vec = w2v_model[word]
                my_file.write('%s %s\n' % (word, ' '.join('%f' % val for val in vec)))

def most_similar(word, w2v_model, anew_dict, weight_dict, top=10):
    sim_array = []; word_array = []

    # get the most similar words from word2vec model 
    similar_words = w2v_model.most_similar(word, topn=top)

    i = 0
    for similar_word in similar_words:
        try:
            diff = weight_dict[word][similar_word[0]]
            sim_array.append([i, diff])
        except:
            sim_array.append([i, 0.0])

        word_array.append(similar_word[0])
        i = i + 1

    sim_array = np.array(sim_array)
    sort_index = sim_array[:, 1].argsort(0)
    new_array = sim_array[sort_index][::-1]

    ret_dict = {}
    for i in range(top):
        word = word_array[int(new_array[i][0])]
        ret_dict[word] = 1. / float(i + 1.)

    return ret_dict

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", "-f", help="pre-trained word embeddings file")
    parser.add_argument("--lexicon", "-l", help="lexicon which provide sentiment intensity")
    parser.add_argument("--top", "-t", help="top-k nearest neighbor", default=10)
    parser.add_argument("--iter", "-i", help="refinement interation", default=100)
    parser.add_argument("--beta", "-b", help="parameter beta", default=0.1)
    parser.add_argument("--gamma", "-g", help="parameter gamma", default=0.1)
    parser.add_argument("--valence", "-v", help="max value of valence", default=9.0)

    args = parser.parse_args()

    valence_max = args.valence
    gamma =args.gamma
    beta = args.beta
    top = args.top
    max_iter = args.iter

    logging.info('loading w2v_model...')
    # w2v_model_file = os.path.join('vector', 'glove.twitter.27B.50d.gensim.txt')
    w2v_model_file = args.filename

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file, binary=False)
    embedding_dim = w2v_model.vector_size
    logging.info('w2v_model loaded!')

    # load lexicon
    logging.info('loading lexicon...')
    # anew_file = os.path.join('lexicon', 'eanew_seed.txt')
    anew_file = args.lexicon
    anew = pd.read_table(anew_file, header=None, sep='\t', quoting=3)
    logging.info('lexicon loaded!')

    logging.info('prepare data...')
    anew_dict = {}
    vector_dict = {}
    for i in range(len(anew[0])):
        try:
            vector_dict[anew[0][i]] = w2v_model[anew[0][i]]
            anew_dict[anew[0][i]] = anew[1][i]
        except:
            continue

    # weight_dict
    logging.info('weight_dict')
    weight_dict = {}
    for i in anew_dict.keys():
        for j in anew_dict.keys():
            weight = valence_max - abs(anew_dict[i] - anew_dict[j])
            update_dict(weight_dict, i, j, weight)

    # weighted matrix
    logging.info('weight_matrix')
    weight_matrix = np.zeros((len(anew_dict), len(anew_dict)))
    for i in range(len(anew_dict.keys())):
        word_i = anew_dict.keys()[i]
        sim_dict = most_similar(word_i, w2v_model, anew_dict, weight_dict, top=top)

        for j in range(len(anew_dict.keys())):
            word_j = anew_dict.keys()[j]
            if word_j in sim_dict.keys():
                weight_matrix[i][j] = sim_dict[word_j]
                weight_matrix[j][i] = sim_dict[word_j]

    # vertex matrix
    logging.info('vertex_matrix')
    vertex_matrix = np.zeros((len(anew_dict.keys()), embedding_dim))
    for i in range(vertex_matrix.shape[0]):
        for j in range(vertex_matrix.shape[1]):
            vector = vector_dict[anew_dict.keys()[i]]
            vertex_matrix[i, j] = vector[j]

    logging.info('weight_matrix shape: ' + str(weight_matrix.shape))
    logging.info('vectex_matrix shape: ' + str(vertex_matrix.shape))
    
    logging.info('starting refinement')
    origin_vertex_matrix = vertex_matrix
    pre_vertex_matrix = vertex_matrix
    pre_distance = 0.0
    diff = 1.0
    num_word = len(anew_dict.keys())

    for iteration in range(max_iter):
        pre_vertex_matrix = vertex_matrix.copy()
        for i in range(num_word):
            denominator = 0.0
            molecule = 0.0
            tmp_vertex = np.zeros((embedding_dim, ))
            weight_sum = 0.0
            for j in range(num_word):
                w_multi_v = weight_matrix[i, j] * pre_vertex_matrix[j]
                weight_sum = weight_sum + weight_matrix[i, j]
                tmp_vertex = tmp_vertex + w_multi_v

            molecule = gamma * pre_vertex_matrix[i] + beta * tmp_vertex
            denominator = gamma + beta * weight_sum
            delta = molecule / denominator
            vertex_matrix[i] = delta
            
            # for k in range(vector_dim):
            #     print(vertex_matrix[i, k], pre_vertex_matrix[i, k], delta[k])

        distance = vertex_matrix - pre_vertex_matrix
        distanceT = distance.T
        value = np.dot(distance, distance.T)

        ec_distance = 0.0
        for i in range(embedding_dim):
            ec_distance = ec_distance + value[i, i]

        diff = abs(ec_distance - pre_distance)
        logging.info('cost: %f' % (diff))
        pre_distance = ec_distance

    refine_vector_file = w2v_model_file + '.refine'
    write_vector(refine_vector_file, w2v_model, vertex_matrix, anew_dict)