import os
import re
import csv

import numpy as np
import pandas as pd
import spacy

import torch
import torchtext
from torchtext import data, vocab



class DatasetReader(object):
    nlp = spacy.load('en_core_web_sm')

    def __init__(self, 
                 features,
                 path,
                 train_filename, valid_filename, test_filename):

        self.text_field = data.Field(sequential=True,
                                    tokenize=self.tokenizer,
                                    #include_lengths=True,
                                    fix_length=16,
                                    batch_first=True,
                                    use_vocab=True)
        
        self.label_field = data.Field(sequential=False, 
                                      use_vocab=True, 
                                      pad_token=None,
                                      batch_first=True, 
                                      unk_token=None)
        
        self.path = path
        self.train_filepath = path + os.sep + train_filename
        self.valid_filepath = path + os.sep + valid_filename
        self.test_filepath  = path + os.sep + test_filename

        self.features = features
        self.target = [col_name for col_name in self.getDatasetColumns() 
                                                if col_name not in self.features]


    def getDatasetIterator(self):
        dataset_fields = list()

        for col in self.features:
            dataset_fields.append((col, self.text_field))
        for col in self.target:
            dataset_fields.append((col, self.label_field))
        
        train_ds, valid_ds = data.TabularDataset.splits(path='./dataset',
                                                        format='tsv',
                                                        train='train.csv',
                                                        validation='valid.csv',
                                                        fields=dataset_fields,
                                                        skip_header=True)
        
        embedding_vector = vocab.Vectors('glove.6B.300d.txt', './pretrained_model/')
        self.text_field.build_vocab(train_ds, vectors=embedding_vector)
        self.label_field.build_vocab(train_ds)

        #print('Training Examples:', vars(train_ds.fields["short_description"]))
        #print('Vocab:', train_ds.fields["short_description"].vocab.vectors)

        train_dl, valid_dl = data.BucketIterator.splits(datasets=(train_ds, valid_ds), 
                                            batch_sizes=(5, 5),  
                                            sort_key=lambda x: len(x.short_description), 
                                            device=-1,
                                            sort_within_batch=True, 
                                            repeat=False)
        
        return train_dl, valid_dl

    
    def getDatasetColumns(self):
        with open(self.train_filepath) as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            col_names = next(tsv_reader)
        
        return col_names


    @classmethod
    def tokenizer(self, text):
        return [w.text.lower() for w in self.nlp(DatasetReader.text_clean(text))]
    
    @staticmethod
    def text_clean(text):
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
        text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
        return text.strip()


class BatchGenerator:
    def __init__(self, dl, x_field, y_field, y_dtype='torch.LongTensor'):
        self.dl, self.x_field, self.y_field = dl, x_field[0], y_field
        self.y_dtype = y_dtype

    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = [getattr(batch, field) for field in self.y_field]
            y = torch.stack(y, dim=1).type(self.y_dtype)
            yield (X,y)



if __name__ == "__main__":
    dataset = DatasetReader(['short_description'], './dataset/', 'train.csv', 'valid.csv', 'test.csv')
    train_dl, valid_dl = dataset.getDatasetIterator()

    train_batch_iter = BatchGenerator(train_dl, dataset.features, dataset.target)
    valid_batch_iter = BatchGenerator(valid_dl, dataset.features, dataset.target)

    batch_train, batch_valid = next(iter(train_batch_iter)), next(iter(valid_batch_iter))
    
    #for batch_train in iter(train_batch_iter):
    #print('Batch Train =', train_batch_iter.size())
    
    