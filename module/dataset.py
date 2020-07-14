import os
from math import floor

import re 

import pickle

import numpy as np 
import pandas as pd
import spacy
from sklearn.preprocessing import LabelBinarizer

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from utils.utils import DataPreprocessing
from utils.constants_reader import Constant

#load contant/cofig reader
constant = Constant()


#Global variable defination
nlp = spacy.load('en_core_web_sm')
word_to_index, emb_matrix, emb_dim = DataPreprocessing.readPretrainedVector()


class TextDataset(Dataset):
    "Create Tensor for Text Classification dataset"

    def __init__(self, csv_filepath, feature, target, transform):
        self.target = target
        self.dataframe = pd.read_csv(csv_filepath, '\t', usecols=[*feature, *target])
        self.target_vector_encoder = self.createTagetVectorEncoder()
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        feature = TextDataset.tokenizer(self.dataframe.iloc[idx, 0])
        target = self.target_vector_encoder.transform([self.dataframe.iloc[idx, 1]])
        target = np.hstack((target, 1 - target))

        sample = {'feature': feature, 'target': target}
        sample = self.transform(sample)

        return sample

    
    def createTagetVectorEncoder(self, label_encoder_filepath=constant.pretrained_config['target_label_filename']):
        label_encoder = LabelBinarizer()
        label_encoder.fit(self.dataframe[self.target[0]])

        with open(label_encoder_filepath, 'wb') as fptr:
            pickle.dump(label_encoder, fptr)
        
        return label_encoder
        

    @staticmethod
    def create_list(string):
        return str(string).split(",")

    @staticmethod 
    def tokenizer(text):
        return np.array( [word_to_index.get(w.text.lower(), 0) for w in nlp(TextDataset.text_clean(text))] )
    

    @staticmethod
    def text_clean(text):
        text = str(text)
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
        text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
        return text.strip()
    
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, max_tensor_length=constant.dataset_config['sequence_length']):
        feature, target = sample['feature'], sample['target']
        
        feature_tensor = torch.from_numpy(feature)
        feature_tensor = feature_tensor[:max_tensor_length]
        pad_length = max(0, max_tensor_length - feature_tensor.size()[0])
        feature_tensor = F.pad(feature_tensor, pad=(0, pad_length), mode='constant', value=0)

        target_tensor = torch.from_numpy(target)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        feature_tensor = feature_tensor.type(torch.LongTensor).to(device)
        target_tensor = target_tensor.type(torch.FloatTensor).to(device)

        return  {'feature': feature_tensor, 'target': target_tensor}


if __name__ == "__main__":
    ''' It's a test execution to check the correctness in creating Tensor from dataset. Please set nrows=50 in pd.read_csv line in init before execution'''


    text_dataset = TextDataset(constant.dataset_config['dataset_file'], constant.dataset_config['features'], constant.dataset_config['labels'], ToTensor())
    
    
    train_dataset, test_dataset = random_split(text_dataset, [40, 10])
    text_dataloader = DataLoader(train_dataset, batch_size=4)

    for i_batch, sample_batched in enumerate(text_dataloader):
        print('Batch:{}'.format(i_batch+1))
        print('Feature:{}, \nTarget:{}'.format(sample_batched['feature'], sample_batched['target']))
