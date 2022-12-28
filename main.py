import os
import pandas as pd
import torch
import torchvision
from torch import nn
import train
from model import AlexNet
from util import rd_csv

if __name__ == '__main__':
    train()