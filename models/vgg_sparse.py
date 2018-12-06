'''VGG_Me in Pytorch.'''
import torch
import torch.nn as nn


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 'Reduce']


class VGG_Sparse(nn.Module):
    def __init__(self):
        super(VGG_Sparse, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(nn.Linear(512, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 10))
        # self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'Reduce':
                layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, padding=0),
                            nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG_Sparse()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
