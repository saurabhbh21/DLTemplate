import os
import pickle

import numpy as np
import spacy

import torch
import torch.nn.functional as F

from module.dataset import TextDataset, ToTensor
from module.model import Model
from utils.constants_reader import Constant

constant = Constant()


class Predict(object):
    '''Predict for list of sentences whether they're actionable or not'''
    
    def __init__(self):
    
        label_encoder_filepath=constant.pretrained_config['target_label_filename']
        with open(label_encoder_filepath, 'rb') as fptr:
            self.label_encoder = pickle.load(fptr)

        self.nn_model = self.loadModel()


        

    def loadModel(self, model_dir=constant.train_config['trained_model_dir'],
                  model_name=constant.predict_config['best_model_name']):
        model_path = model_dir + os.sep + model_name
        
        classifier_model = Model(len(self.label_encoder.classes_))
        nn_model = classifier_model.createModel()
        
        nn_model.load_state_dict(torch.load(model_path))
        nn_model.eval()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        return nn_model.to(device)


    def createTensor(self, sentences, max_tensor_length=constant.dataset_config['sequence_length']):
        feature = np.zeros((len(sentences), max_tensor_length), dtype=np.int32)

        for i in range(len(sentences)): 
            embedding_index_vector = TextDataset.tokenizer(sentences[i])
            feature[i, :embedding_index_vector.shape[0]] = embedding_index_vector   
        
        feature_tensor = torch.from_numpy(feature)
        feature_tensor = feature_tensor[:max_tensor_length]
        pad_length = max(0, max_tensor_length - feature_tensor.size()[0])
        feature_tensor = F.pad(feature_tensor, pad=(0, pad_length), mode='constant', value=0)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        return feature_tensor.type(torch.LongTensor).to(device)


    def getLabelNames(self, predicted_labels):
        label_names = self.label_encoder.inverse_transform(predicted_labels)
        return label_names.tolist()
    
    
    def predict(self, sentences):
        tensor = self.createTensor(sentences)

        predicted_logits = self.nn_model(tensor)
        predicted_labels = torch.argmax(predicted_logits, dim=1, keepdim=True).cpu().numpy()
        label_names = self.getLabelNames(predicted_labels)

        return list(zip(sentences, label_names))
        

    

if __name__ == "__main__":
    sentence = ['Saurabh', 'Please complete as soon as possible.']
    prediction = Predict()
    predicted_labels = prediction.predict(sentence)
    
    print('Sentence-Label Pair: {}'.format(predicted_labels) )
    