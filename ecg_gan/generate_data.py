import os
import time
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam

from gan import Generator, Discriminator
from dataset import ECGDataset, get_dataloader
from config import Config


class DataGenerator:
    def __init__(
            self,
            generator_state_dict_path
    ):
        generator = Generator()
        generator.load_state_dict(torch.load(generator_state_dict_path))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.netG = generator.to(self.device)

    def generate(self, num_data_points):
        noise = torch.randn(num_data_points, 1, 187, device=self.device)
        fake = self.netG(noise)
        generated_data = fake.detach().cpu().squeeze(1).numpy()[:].transpose()
        return generated_data


if __name__ == '__main__':
    config = Config()

