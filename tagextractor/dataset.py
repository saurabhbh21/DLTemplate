import os
from math import floor

import numpy as np 
import pandas as pd


class Dataset(object):
    def __init__(self, dir_path, file_name, feature_cols, target_cols):
        self.input_file_path = dir_path + os.sep + file_name

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.df = self.createCategorizedDataframe([*self.feature_cols, *self.target_cols])
        self.train_df, self.valid_df, self.test_df = self.splitAndSave()
        
    
    def createCategorizedDataframe(self, col_names):
        dataframe = pd.read_csv(self.input_file_path, usecols=col_names, delimiter='\t')      
        categorized_df =  dataframe.join(dataframe.pop('category_list').str.get_dummies(sep=","))
        
        return categorized_df
    
    def splitAndSave(self):
        output_dir = os.path.dirname(self.input_file_path)
        output_train_path = output_dir + os.sep + 'train.csv'
        output_valid_path = output_dir + os.sep + 'valid.csv'
        output_test_path = output_dir + os.sep + 'test.csv'
        
        train_df = self.df.iloc[:floor(len(self.df)*0.8)]
        valid_df = self.df.iloc[floor(len(self.df)*0.8) : floor(len(self.df)*0.9)]
        test_df = self.df.iloc[floor(len(self.df)*0.9):]

        train_df.to_csv(output_train_path, sep='\t', index=False)
        valid_df.to_csv(output_valid_path, sep='\t', index=False)
        test_df.to_csv(output_test_path, sep='\t', index=False)

        return train_df, valid_df, test_df
    

if __name__ == "__main__":
    dir_path = os.getcwd() + os.sep + 'dataset'
    file_name = 'orgs.csv'
    dataset = Dataset(dir_path, file_name, feature_cols=['short_description'], target_cols=['category_list'])
    print(len(dataset.train_df), len(dataset.valid_df), len(dataset.test_df))
        