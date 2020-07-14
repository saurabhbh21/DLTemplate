import torch 
import torch.nn as nn
from torch.autograd import Variable

from utils.utils import DataPreprocessing
from utils.constants_reader import Constant

constant = Constant()

class NeuralNetwork(nn.Module):
    '''
    Defined Neural Network Architecture for Text Classification
    '''

    def __init__(self, hidden_dim, layer_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        #create embedding matrix
        self.embedding, embedding_dim = self.createEmbeddingLayer()

        #LSTM followed by fully connected NN
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(0.3), #30 % probability 
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.Dropout(0.3), #30 % probability 
            nn.ReLU(),
            #nn.Linear(512, output_dim),
            nn.Linear(128, output_dim),

            nn.Softmax()
        )
        


    def forward(self, x):
        # Initialize hidden state with zeros   
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        hidden = ( torch.randn(self.layer_dim, x.size(0), self.hidden_dim).to(device),
                   torch.randn(self.layer_dim, x.size(0), self.hidden_dim).to(device) )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            
        # one time-step computation of RNN
        out, _ = self.rnn(self.embedding(x), hidden)
        out = out[:, -1, :]
        out = self.classifier(out) 

        return out

    
    def createEmbeddingLayer(self, non_trainable=True):
        _, emb_matrix, _ = DataPreprocessing.readPretrainedVector()
        embedding_matrix = torch.tensor(emb_matrix)
        num_embeddings, embedding_dim = embedding_matrix.size()
        
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': embedding_matrix})
        
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, embedding_dim

 


class Model(NeuralNetwork):
    '''
    Create a model(instance) of the Neural Network 
    '''

    def __init__(self,  
                       output_dim,
                       hidden_dim=constant.model_config['hidden_dim'], 
                       layer_dim=constant.model_config['layer_dim'], 
                       lr = constant.model_config['learning_rate']):
        
        super(Model, self).__init__(hidden_dim, layer_dim, output_dim)
        #neural network hyper-parameter
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.learning_rate = lr
        
        #neural network object on GPU(if available)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.nn_model = self.createModel().to(device)
        
        
    def createModel(self):
        return NeuralNetwork(self.hidden_dim, self.layer_dim, self.output_dim)
    
    def error_optimizer(self):
        error = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)
        return error, optimizer
