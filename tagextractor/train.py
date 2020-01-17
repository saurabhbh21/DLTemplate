import torch

from tagextractor.datareader import DatasetReader, BatchGenerator
from tagextractor.model import Model

class Train(object):
    def __init__(self,batch_size=5, num_epochs=5):

        #traing parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        #data-loader for training in pytorch dataloader format
        self.train_batch, self.valid_batch = self.loadData()

        #NN Model
        self.classifier_model = Model()

        #Metric parameters for evaluation
        self.loss_list = list()
        self.iteration_list = list()
        self.accuracy_list = list()
        self.count = 0

        
    def loadData(self):
        dataset = DatasetReader(['short_description'], './dataset/', 'train.csv', 'valid.csv', 'test.csv')
        train_dl, valid_dl = dataset.getDatasetIterator()

        train_batch_iter = BatchGenerator(train_dl, dataset.features, dataset.target)
        valid_batch_iter = BatchGenerator(valid_dl, dataset.features, dataset.target)

        return train_batch_iter, valid_batch_iter


    def train(self):
        for epoch in range(self.num_epochs):
            for index, (data, label) in enumerate(self.train_batch):
            
                #load error and optimizer function
                error, optimizer = self.classifier_model.error_optimizer()

                #clear gradient
                optimizer.zero_grad()

                #forward propagation
                batch_output = self.classifier_model.nn_model(data)

                #calculate loss
                loss = error(batch_output, label)

                #claculate gradient and update weight
                loss.backward()
                optimizer.step()

                self.count += 1
        
                if self.count % 2 == 0:
                             
                    correct = 0
                    total = 0
                    
                    # Iterate through test dataset
                    for (test_data, test_label) in self.train_batch:
                        test_output = self.classifier_model.nn_model(test_data)

                        print('Test Output =', test_output)



                

        

if __name__ == "__main__":
    training = Train()
    training.train()
        