import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

from tagextractor.dataset import TextDataset, ToTensor
from tagextractor.model import Model
from tagextractor.utils import Logger, EvaluationMetric
from tagextractor.constants_reader import Constant

constant = Constant()

class Train(object):
    def __init__(self,
                 batch_size=constant.train_config['train_batch_size'], 
                 num_epochs=constant.train_config['num_epochs'], 
                 dataset_path=constant.dataset_config['dataset_file'],
                 features=constant.dataset_config['features'], 
                 labels=constant.dataset_config['labels']):

        #traing parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        #data-loader for training in pytorch dataloader format
        self.train_batch, self.valid_batch, self.target_num_classes = self.loadData(dataset_path, features, labels)

        #NN Model functions
        self.classifier_model = Model(self.target_num_classes)
        self.error, self.optimizer = self.classifier_model.error_optimizer()
        
        #logging description
        self.model_name = constant.model_config['model_name']

        
    def loadData(self, dataset_path, features, labels, test_split=constant.train_config['test_split_ratio']):
        text_dataset = TextDataset(dataset_path, features, labels, ToTensor())
        target_num_classes = len(text_dataset.target_vector_encoder.classes_)
        
        test_size = int(math.floor(len(text_dataset)*test_split))
        train_size = int(len(text_dataset) - test_size)
        train_ds, test_ds = random_split(text_dataset, [train_size, test_size])
        
        train_dl = DataLoader(train_ds, batch_size=constant.train_config['train_batch_size'])
        valid_dl = DataLoader(test_ds, batch_size=constant.train_config['valid_batch_size'])

    
        return train_dl, valid_dl, target_num_classes


    def train(self, 
             model_dir=constant.train_config['trained_model_dir'], 
             model_name=constant.predict_config['best_model_name']):

        iteration_step = 0
        logger = Logger(self.model_name)

        model_path = model_dir + os.sep + model_name
        self.classifier_model.nn_model.load_state_dict(torch.load(model_path))
        self.classifier_model.nn_model.eval()

        start_idx_epoch = 43
        for epoch in range(start_idx_epoch, start_idx_epoch+self.num_epochs): 
            print('Executing Epoch: {}'.format(epoch))
            
            #execute each batch
            for sample in iter(self.train_batch):
                #extract data and label
                data = sample['feature']
                label = sample['target']

                #clear gradient
                self.optimizer.zero_grad()

                #forward propagation
                batch_output = self.classifier_model.nn_model(data)
                
                #calculate loss
                loss = self.error(batch_output, label)

                #claculate gradient and update weight
                loss.backward()
                self.optimizer.step()
                                                        
                # Find metrics on validation dataset
                iteration_step += self.batch_size
                
            eval_metric = EvaluationMetric(self.batch_size, self.target_num_classes, self.classifier_model.nn_model)
            accuracy_train = eval_metric.calculateEvaluationMetric(self.train_batch, loss.data)
            accuracy_valid = eval_metric.calculateEvaluationMetric(self.valid_batch, loss.data)
            print('Epoch: {}   loss: {}   Test Accuracy: {},  Train Accuracy: {}'.format(epoch, loss.data, accuracy_valid, accuracy_train))

            #log the metric in graph with tensorboard
            logger.log(accuracy_train, accuracy_valid, loss, iteration_step)
                
            #save the model weights
            model_filepath = model_dir + os.sep + 'weight_epoch-{}_loss-{}'.format(epoch, loss.data)
            torch.save(self.classifier_model.nn_model.state_dict(), model_filepath)
        
        logger.close()


    
    

    

if __name__ == "__main__":
    training = Train()
    training.train()
        