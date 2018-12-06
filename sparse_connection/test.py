'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse
import sys
sys.path.insert(0, "..")

from models import VGG_Me

from utils import progress_bar, compute_param_numbers

print('s')