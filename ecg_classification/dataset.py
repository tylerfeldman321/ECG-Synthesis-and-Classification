import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from config import Config


class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.FloatTensor(np.array([signal.values]))
        target = torch.LongTensor(np.array([self.df.loc[idx, 'class']]))
        return signal, target

    def __len__(self):
        return len(self.df)


def get_dataloader(phase: str, batch_size: int = 96) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        pahse: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    df = pd.read_csv(Config.train_csv_path)
    test_df = pd.read_csv(Config.test_csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=Config.seed, stratify=df['label']
    )
    train_df, val_df, test_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    if phase == 'train':
        df = train_df
    elif phase == 'val':
        df = val_df
    elif phase == 'test':
        df = test_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2)
    return dataloader


if __name__ == '__main__':
    train_dataloader = get_dataloader(phase='train', batch_size=96)
    val_dataloader = get_dataloader(phase='val', batch_size=96)
    test_dataloader = get_dataloader(phase='test', batch_size=96)
