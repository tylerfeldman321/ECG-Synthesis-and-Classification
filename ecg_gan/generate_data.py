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
            generator_state_dict_path,
            subject_id
    ):
        generator = Generator()
        generator.load_state_dict(torch.load(generator_state_dict_path))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.netG = generator.to(self.device)
        self.subject_id = subject_id

    def generate(self, num_data_points):
        noise = torch.randn(num_data_points, 1, 187, device=self.device)
        fake = self.netG(noise)
        generated_data = fake.detach().cpu().squeeze(1).numpy()[:]
        subject_data = np.array([self.subject_id] * generated_data.shape[0])
        subject_data = np.expand_dims(subject_data, axis=0).transpose()
        generated_data = np.hstack((generated_data, subject_data))
        generated_data_df = pd.DataFrame(generated_data)
        return generated_data_df


class FakeDataset:
    def __init__(self, generator_state_dict_paths, subject_ids):
        self.data_generators = [DataGenerator(generator_state_dict_paths[i], subject_ids[i]) for i in
                                range(len(generator_state_dict_paths))]

    def generate(self):
        dataset = pd.DataFrame()
        for data_generator in self.data_generators:
            subject_data = data_generator.generate(5000)
            dataset = pd.concat([dataset, subject_data], ignore_index=True)
        return dataset


if __name__ == '__main__':
    config = Config()
    subject_id_list = [2, 3, 4, 5, 7, 8, 10, 13, 14, 15, 16]
    # subject_id_list = [2]
    state_dict_paths = [f's{id}-generator-epoch-500.pth' for id in subject_id_list]
    # print(state_dict_paths)
    # dg = DataGenerator('generator-epoch-500.pth', 0)
    # df = dg.generate(100)
    # print(df.head(), df.info(), df.shape)

    fake_dataset = FakeDataset(state_dict_paths, subject_id_list)
    df = fake_dataset.generate()
    df = df.sample(frac=1)
    df.to_csv('wesad_train.csv', header=False, index=False)
