import torch
import torch.nn as nn


class FCNN(nn.Module):
    """

    Input:
        input_size: feature size , int
        hidden_size : hidden size , tuple:[200,200,...]
        n_classes : int , 2
        p : dropout ,probability of an element to be zeroed. Default: 0.5
        device : cpu or gpu

    Output:
        out: (n_samples)

    Pararmetes:
        n_classes: number of classes

    """

    def __init__(self, input_size, hidden_size, n_classes=2, p=0.5, device="cpu"):
        super(FCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.device = device

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size[0]))
        self.layers.append(nn.Dropout(p))

        for i in range(1, len(self.hidden_size)):
            self.layers.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
            self.layers.append(nn.Dropout(p))

        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_size[-1], self.n_classes))
        print("model is {}".format(self))

    def forward(self, x):
        return self.layers(torch.tensor(x, dtype=torch.float32))
