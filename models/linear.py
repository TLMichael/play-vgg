'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn



class FullLinear(nn.Module):
    def __init__(self):
        super(FullLinear, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(3072, 4096), 
                                        # nn.ReLU(),
                                        nn.Linear(4096, 1024), 
                                        nn.Linear(1024, 10))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out



def test():
    net = FullLinear()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
