import pandas as pd 
from utils.constants_reader import Constant


constant = Constant()

class CreateDataset(object):

    def __init__(self):
        '''Initialisation'''
        pass 


    def writeToCSV(self, urls, features, target, output_path=constant.dataset_config['dataset_file']):
        '''Create dataset with train.csv'''
        pass 


    @staticmethod
    def extractTextContent():
        ''' Do the pre-processing to extract the content for training dataset'''
        pass 
        

if __name__ == "__main__":
    dataset = CreateDataset()
    