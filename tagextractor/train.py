import os

import torch

from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import DataLoader, random_split

from tagextractor.dataset import TextDataset, ToTensor
from tagextractor.model import Model
from tagextractor.constants_reader import Constant

constant = Constant()

class Train(object):
    def __init__(self,batch_size=5, num_epochs=5, 
                 dataset_path=constant.dataset_config['dataset_file'],
                 features=constant.dataset_config['features'], 
                 labels=constant.dataset_config['labels']):

        #traing parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        #data-loader for training in pytorch dataloader format
        self.train_batch, self.valid_batch = self.loadData(dataset_path, features, labels)

        #NN Model functions
        self.classifier_model = Model()
        self.error, self.optimizer = self.classifier_model.error_optimizer()

        #Metric parameters for evaluation
        self.loss_list = list()
        self.iteration_list = list()
        self.accuracy_list = list()
        

        
    def loadData(self, dataset_path, features, labels):
        text_dataset = TextDataset(dataset_path, features, labels, ToTensor())
        train_ds, test_ds = random_split(text_dataset, [40, 10])
        
        train_dl = DataLoader(train_ds, batch_size=5)
        valid_dl = DataLoader(test_ds, batch_size=5)

    
        return train_dl, valid_dl


    def train(self, model_dir=constant.train_config['trained_model_dir']):
        for epoch in range(self.num_epochs): 
            print('Executing Epoch: {}'.format(epoch))

            #execute each batch
            for batch_idx, sample in enumerate(iter(self.train_batch)):
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
                accuracy = self.validation_data_metric(loss.data)
                print('Batch: {}   loss: {}   Accuracy: {}'.format(batch_idx, loss.data, accuracy))


            #save the model weights
            model_filepath = model_dir + os.sep + 'weight_epoch-{}_loss-{}'.format(epoch, loss.data)
            torch.save(self.classifier_model.nn_model.state_dict(), model_filepath)


    def validation_data_metric(self, loss):
        "ToDo: Add tensorboard and better code for calculating validation"
        actual = None
        predicted = None

        for test_sample in iter(self.valid_batch):
            test_data = test_sample['feature'] 
            test_label = test_sample['target']

            predicted_logits = self.classifier_model.nn_model(test_data)
            
            predicted = torch.clamp(torch.round(predicted_logits.data), 0, 1).numpy().astype(int)
            actual = test_label.squeeze(1).numpy().astype(int)
                
        accuracy =  f1_score(actual, predicted, average='macro')

        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)

        return accuracy
    

    

if __name__ == "__main__":
    training = Train()
    training.train()
        