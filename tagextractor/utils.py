import numpy as np
from tagextractor.constants_reader import Constant

constant = Constant()

class PretrainedVector(object):

    @staticmethod
    def readPretrainedVector(glove_path=constant.pretrained_config['embedding_filename']):
        with open(glove_path, 'r') as f:
            words = set()
            word_to_vec_map = {}
            
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype = np.float32)   
            
            i = 1
            word_to_index = {}
            index_to_word = {}
            
            for w in sorted(words):
                word_to_index[w] = i
                index_to_word[i] = w
                i = i + 1
            
            vocab_len = len(word_to_index) + 1                  
            emb_dim = word_to_vec_map["cucumber"].shape[0]     
            emb_matrix = np.zeros((vocab_len, emb_dim))
    
            for word, index in word_to_index.items():
                emb_matrix[index, :] = word_to_vec_map[word]
            
        return word_to_index, emb_matrix, emb_dim