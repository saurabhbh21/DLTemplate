import email

import numpy as np
import torch 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.constants_reader import Constant

constant = Constant()

class DataPreprocessing(object):
    def __init__(self):
        pass

    @staticmethod
    def extractMessageFromEmail(email_message):
        email_body = email.message_from_string(email_message).get_payload()
        return email_body

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


    def log(self, accuracy_train, accuracy_valid, training_loss, validation_loss, iteration_step):
        if isinstance(accuracy_train, torch.autograd.Variable):
            accuracy_train = accuracy_train.data.cpu().numpy()

        if isinstance(accuracy_valid, torch.autograd.Variable):
            accuracy_valid = accuracy_valid.data.cpu().numpy()
        
        if isinstance(training_loss, torch.autograd.Variable):
            training_loss = training_loss.data.cpu().numpy()

        if isinstance(validation_loss, torch.autograd.Variable):
            validation_loss = validation_loss.data.cpu().numpy()


        if isinstance(iteration_step, torch.autograd.Variable):
            iteration_step = iteration_step.data.cpu().numpy()

        self.writer.add_scalars('{}/Loss'.format(self.comment), {'train_loss': training_loss, 'valid_loss': validation_loss}, iteration_step)
        self.writer.add_scalars('{}/F1-Score'.format(self.comment), {'train_f1': accuracy_train, 'valid_f1': accuracy_valid}, iteration_step)


    def close(self):
        self.writer.close()
    
    

class EvaluationMetric(object):
    def __init__(self, target_num_classes):
        self.target_num_classes = target_num_classes
        
        

    def calculateLoss(self, dataset, batch_size, trained_model, loss_function):
        
        running_loss = 0.0
        num_batches = len(dataset)

        for test_sample in iter(dataset):
            test_feature = test_sample['feature']
            actual_label = test_sample['target']
            predicted_labels = trained_model(test_feature)

            loss = loss_function(predicted_labels, actual_label[:, 0, :])
            running_loss += loss.item()

        return running_loss/(num_batches*batch_size)


    
    def calculateEvaluationMetric(self, batch, batch_size, trained_model):
        actual = torch.zeros([len(batch)*batch_size, self.target_num_classes])
        predicted = torch.zeros([len(batch)*batch_size, self.target_num_classes])
        index = 0

        for test_sample in iter(batch):
            test_data = test_sample['feature'] 
            test_label = test_sample['target']

            predicted_logits = trained_model(test_data)
            num_instnaces = predicted_logits.size()[0]


            predicted[index: index+num_instnaces] = predicted_logits.data
            actual[index: index+num_instnaces] = test_label.squeeze(1)
            
            index = index + num_instnaces


        precision_metric = precision_score(actual.cpu(), predicted.cpu()>0.5, average='macro')
        recall_metric = recall_score(actual.cpu(), predicted.cpu()>0.5, average='macro')
        f1_metric = f1_score(actual.cpu(), predicted.cpu()>0.5, average='macro')


        return precision_metric, recall_metric, f1_metric



