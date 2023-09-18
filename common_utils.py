### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.io import wavfile as wav
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            
            nn.Dropout(0.2),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )

    # YOUR CODE HERE
    def forward(self, x):
        return self.mlp_stack(x)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels)

        self.features = torch.tensor(features, dtype=torch.float32)
        # model output shape is (X, 1), so we need to reshape our labels from 1D to 2D
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1) 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = self.labels[idx]
        features = self.features[idx]

        return features, label

def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test):
    train_data = CustomDataset(X_train_scaled, y_train)
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

    test_data = CustomDataset(X_test_scaled, y_test)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True)

    return train_dataloader, test_dataloader

loss_fn = nn.BCELoss()