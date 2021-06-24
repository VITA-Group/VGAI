import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd

class ReluNet(torch.nn.Module):
    def __init__(self, filter_length=3):
        super(ReluNet, self).__init__()
        n_feature = filter_length * 6       
        layer_sizes = [n_feature, 1024, 1024, 2]
        self.n_layers = len(layer_sizes) - 1

        # hidden layers
        self.hidden = []
        for n in range(0, self.n_layers - 1):
            self.hidden.append(torch.nn.Linear(layer_sizes[n], layer_sizes[n + 1]))
        self.hidden = torch.nn.ModuleList(self.hidden)

        # output layer
        self.predict = torch.nn.Linear(layer_sizes[self.n_layers - 1], layer_sizes[self.n_layers])

    def forward(self, x):
        for n in range(0, self.n_layers - 1):
            x = torch.nn.functional.relu((self.hidden[n](x)))
        return self.predict(x)

def loc_dagnn(K=3, **kwargs):
    # n = 6
    model = ReluNet(filter_length=K)
    return model




def main():
    print('test model')
    x = torch.rand(128, 18)
    model = loc_dagnn(K=3)
    u = model(x)
    print(u.shape)

if __name__ == "__main__":
    main()
