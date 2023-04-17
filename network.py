import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 128, bias=True),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 128, bias=False),
                                    nn.Tanh(),
                                    nn.Linear(128, 3, bias=False),
                                    nn.Sigmoid())


    def forward(self, x):
        return self.layers(x)
