import numpy as np

import torch 
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score

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




class Logger(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.comment = '{}'.format(model_name)
        self.writer = SummaryWriter(comment=self.comment)


    def log(self, accuracy_train, accuracy_valid, loss, iteration_step):
        if isinstance(accuracy_train, torch.autograd.Variable):
            accuracy_train = accuracy_train.data.cpu().numpy()
        if isinstance(accuracy_valid, torch.autograd.Variable):
            accuracy_valid = accuracy_valid.data.cpu().numpy()
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.data.cpu().numpy()
        if isinstance(iteration_step, torch.autograd.Variable):
            iteration_step = iteration_step.data.cpu().numpy()

        self.writer.add_scalars('{}/accuracy'.format(self.comment), {'train_acc': accuracy_train, 'valid_acc': accuracy_valid}, iteration_step)
        self.writer.add_scalar('{}/loss'.format(self.comment), loss, iteration_step)


    def close(self):
        self.writer.close()
    
    

class EvaluationMetric(object):
    def __init__(self, batch_size, target_num_classes, trained_model):
        self.batch_size = batch_size
        self.target_num_classes = target_num_classes
        self.trained_model = trained_model

    
    def calculateEvaluationMetric(self, batch, loss):    
        actual = torch.zeros([len(batch)*self.batch_size, self.target_num_classes], dtype=torch.int8)
        predicted = torch.zeros([len(batch)*self.batch_size, self.target_num_classes], dtype=torch.int8)
        index = 0

        for test_sample in iter(batch):
            test_data = test_sample['feature'] 
            test_label = test_sample['target']

            predicted_logits = self.trained_model(test_data)
            num_instnaces = predicted_logits.size()[0]

            predicted[index: index+num_instnaces] = torch.clamp(torch.round(predicted_logits.data), 0, 1)
            actual[index: index+num_instnaces] = test_label.squeeze(1)
            
            index = index + num_instnaces
   
        accuracy =  f1_score(actual, predicted, average='micro')

        return accuracy
