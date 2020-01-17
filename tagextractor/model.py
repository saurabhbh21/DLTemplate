import torch 
import torch.nn as nn
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        #RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)

        #Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    

    def forward(self, x):
        # Initialize hidden state with zeros
        import pdb;pdb.set_trace()
        print('Shape=', x.size())
        hidden = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            
        # # One time step
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out) 

        return out, hidden


class Model(NeuralNetwork):
    def __init__(self,  
                       input_dim=16, hidden_dim=100, layer_dim=1, output_dim=78, lr = 0.05):
        
        super(Model, self).__init__(input_dim, hidden_dim, layer_dim, output_dim)
        #neural network hyper-parameter
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.learning_rate = lr

        #nueral network object
        self.nn_model = self.createModel()
        
        
    def createModel(self):
        return NeuralNetwork(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim)
    
    def error_optimizer(self):
        error = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)
        return error, optimizer




if __name__ == "__main__":
    model = Model()
        