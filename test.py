import pandas as pd
import numpy as np
import pickle

from utils.constants_reader import Constant
from module.dataset import TextDataset, ToTensor

constant = Constant()

# csv_filepath = "./dataset/cb_tags.csv"
# dataframe = pd.read_csv(csv_filepath, "\t", nrows=100000)
# # counts = dataframe['short_description'].str.split().apply(len).value_counts()

# counts = [ len(str(desc).split()) for desc in dataframe['short_description'].values ]
# print(sum(counts)/len(counts))


if __name__ == "__main__":
    text_dataset = TextDataset(constant.dataset_config['dataset_file'], constant.dataset_config['features'], constant.dataset_config['labels'], ToTensor())
    vector_label = text_dataset.target_vector_encoder.transform(text_dataset.dataframe.iloc[:, 1].values)
    count_labels = np.sum(vector_label, axis=0)
    print('Number of Instances for Label: {} \nshape of vector: {}'.format(count_labels, count_labels.shape) )
    print('Max: {} \nMin: {} \nAverage: {} \nMedian:{}'.format(np.max(count_labels), np.min(count_labels), np.mean(count_labels), np.median(count_labels)) )