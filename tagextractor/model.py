import torch 
import torch.nn as nn
from torch.autograd import Variable

from tagextractor.utils import PretrainedVector
from tagextractor.constants_reader import Constant

constant = Constant()

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim, layer_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        #create embedding matrix
        self.embedding, embedding_dim = self.createEmbeddingLayer()

        #RNN followed bu fully connected NN
        self.rnn = nn.RNN(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Initialize hidden state with zeros   
        hidden = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            
        # one time-step computation of RNN
        out, _ = self.rnn(self.embedding(x), hidden)
        out = out[:, -1, :]
        out = self.fc(out) 
        out = self.sigmoid(out)

        return out

    
    def createEmbeddingLayer(self, non_trainable=True):
        _, emb_matrix, _ = PretrainedVector.readPretrainedVector()
        embedding_matrix = torch.tensor(emb_matrix)
        num_embeddings, embedding_dim = embedding_matrix.size()
        
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': embedding_matrix})
        
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, embedding_dim

 


class Model(NeuralNetwork):
    def __init__(self,  
                       hidden_dim=constant.model_config['hidden_dim'], 
                       layer_dim=constant.model_config['layer_dim'], 
                       output_dim=79,
                       lr = constant.model_config['learning_rate']):
        
        super(Model, self).__init__(hidden_dim, layer_dim, output_dim)
        #neural network hyper-parameter
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.learning_rate = lr
        
        #nueral network object
        self.nn_model = self.createModel()
        
        
    def createModel(self):
        return NeuralNetwork(self.hidden_dim, self.layer_dim, self.output_dim)
    
    def error_optimizer(self):
        error = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)
        return error, optimizer




if __name__ == "__main__":
    model = Model()
        