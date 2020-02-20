import os
import pickle

import numpy as np
import spacy

import torch
import torch.nn.functional as F

from tagextractor.dataset import TextDataset, ToTensor
from tagextractor.model import Model
from tagextractor.constants_reader import Constant

constant = Constant()


class Predict(object):
    def __init__(self, description):
        self.description = description

        label_encoder_filepath=constant.pretrained_config['target_label_filename']
        with open(label_encoder_filepath, 'rb') as fptr:
            self.label_encoder = pickle.load(fptr)
        

    def loadModel(self, model_dir=constant.train_config['trained_model_dir'],
                  model_name=constant.predict_config['best_model_name']):
        model_path = model_dir + os.sep + model_name
        
        classifier_model = Model(len(self.label_encoder.classes_))
        nn_model = classifier_model.createModel()
        
        nn_model.load_state_dict(torch.load(model_path))
        nn_model.eval()

        return nn_model


    def createTensor(self, max_tensor_length=constant.dataset_config['sequence_length']):
        feature = TextDataset.tokenizer(self.description)

        feature_tensor = torch.from_numpy(feature)
        feature_tensor = feature_tensor[:max_tensor_length]
        pad_length = max(0, max_tensor_length - feature_tensor.size()[0])
        feature_tensor = F.pad(feature_tensor, pad=(0, pad_length), mode='constant', value=0)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        return feature_tensor.type(torch.LongTensor).to(device)


    def predict(self):
        tensor = self.createTensor()
        nn_model = self.loadModel()

        predicted_logits = nn_model(tensor.unsqueeze(0))
        predicted_labels = torch.as_tensor( predicted_logits > 0.5, dtype=torch.int8).numpy().astype(int)

        return predicted_logits, predicted_labels
        

    def getLabelNames(self, predicted_labels):
        label_names = self.label_encoder.inverse_transform(predicted_labels)
        return label_names[0]



if __name__ == "__main__":
    desc = 'Zoho is an Indian web-based online office suite containing word processing, spreadsheets, presentations, databases, note-taking, wikis, web conferencing, customer relationship management, project management, invoicing, and other applications developed by Zoho Corporation.'
    prediction = Predict(desc)
    predicted_result = prediction.predict()
    label_names = prediction.getLabelNames(predicted_result[1])

    print('Label Names: {}  \n num labels: {}'.format(label_names, len(label_names)) )
    