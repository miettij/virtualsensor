import torch
import torch.nn as nn

class LSTM_layers(nn.Module):
    """LSTM PARAMETERS:
    - input_size = dimensionality of input sequences, give 4 if using 4000,4 input_seq
    - hidden_size = dimensionality of hidden_size, not yet totally clear how it affects
    - num_layers = number of recurrence of a LSTM cell
    - output_size = dimensionality of output sequence
    """

    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.2, output_size = 1, bidirectional = False, batch_first = False):

        super(LSTM_layers, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2 if bidirectional else 1
        

        self.lstm1 = nn.LSTM(
                        input_size = self.input_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        dropout = dropout,
                        batch_first = batch_first,
                        bidirectional = bidirectional)

        self.linear = nn.Linear(self.hidden_size*self.num_directions, self.output_size)

    def forward(self, input_seq, state):
        output, (hn,cn) = self.lstm1(input_seq, state)
        return self.linear(output)

    def init_hidden(self, batch_size, var_params):
        """Try encoding the roll running params here when training with
        multiple rolls"""
        h0 = torch.zeros((self.num_layers*self.num_directions, batch_size, self.hidden_size))

        return h0

    def init_cell(self, batch_size):
        return torch.zeros((self.num_layers*self.num_directions, batch_size, self.hidden_size))
