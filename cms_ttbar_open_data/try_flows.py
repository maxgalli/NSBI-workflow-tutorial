import os, sys, importlib
sys.path.append('../')

import common_utils
from common_utils import plotting, training
from common_utils.training import TrainEvaluate_NN, TrainEvaluatePreselNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from tensorflow.keras.optimizers import Nadam
import mplhep as hep
import matplotlib.pyplot as plt
import pickle
import zuko
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
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

class CustomDataset(Dataset):
    def __init__(self, pd_dataset, columns, device=None):
        self.data = pd_dataset[columns].values
        self.columns = columns
        #self.weights = pd_dataset["weights_normed"].values
        self.weights = pd_dataset["weights"].values
        if device is not None:
            self.data = torch.tensor(self.data, dtype=torch.float32).to(device)
            self.weights = torch.tensor(self.weights, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.weights[idx]

def plot_loss(train_history, test_history, directory, process):
    fig, ax = plt.subplots()
    ax.plot(train_history, label="Train")
    ax.plot(test_history, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(directory + f"{process}_loss.png")
    plt.close(fig)

def evaluate_flow(flow, test_loader, directory, columns, process):
    # plot distribution sampled from the flow and compare to the original distribution
    # test_dataset is a pandas
    flow = flow.to("cpu")
    with torch.no_grad():
        sample_list = []
        data_list = []
        weights_list = []
        for data, weights in test_loader:
            data = data.detach().cpu().numpy()
            weights = weights.detach().cpu().numpy()
            samples = flow().sample((len(data),))
            sample = samples.reshape(-1, len(columns))
            sample_list.append(samples)
            data_list.append(data)
            weights_list.append(weights)
            #print("Length of data: ", len(data))
            #print("Length of sample: ", len(sample))
            #print("Length of weights: ", len(weights))
    sample = np.concatenate(sample_list)
    data = np.concatenate(data_list)
    weights = np.concatenate(weights_list)

    # plot
    for i, column in enumerate(columns):
        fig, ax = plt.subplots()
        ax.hist(data[:, i], bins=100, histtype="step", label="Test data", weights=weights, density=False)
        ax.hist(sample[:, i], bins=100, histtype="step", label="Flow", weights=weights, density=False)
        ax.set_xlabel(column)
        ax.legend()
        fig.savefig(directory + f"flow_{process}_{column}.png")
        plt.close(fig)


if __name__ == '__main__':
    # check with torch if on GPU
    print("CUDA available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    #processes = ["ttbar", "wjets", "single_top_t_chan"]
    processes = ["ttbar"]
    #processes = ["wjets", "single_top_t_chan"]
    for process in processes:
        #process = "wjets"
        #process = "ttbar"
        #process = "single_top_t_chan"

        saved_data='./cached_data/'
        dataset = pd.read_hdf(saved_data + "dataset_preselected_ttbar.h5", "dataset")

        dataset = dataset[dataset["type"] == process]
        print(len(dataset))

        # keep only 30% of events if process is ttbar
        if process == "ttbar":
            dataset = dataset.sample(frac=0.3, random_state=42)

        columns = ['log_lepton_pt', 'log_H_T', 'lepton_eta', 'lepton_phi']

        # scale to (-5, 5) where the spline is defined
        for column in columns:
            scaler = MinMaxScaler(feature_range=(-5, 5))
            dataset[column] = scaler.fit_transform(dataset[[column]]).flatten()

        flow = zuko.flows.NSF(features=len(columns), context=0, transforms=3)
        flow.to(device)

        epochs = 400
        #early_stopper = EarlyStopper(patience = 15, min_delta=0.000)
        batch_size = 256
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
        train_dataset = CustomDataset(train_dataset, columns, device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = CustomDataset(val_dataset, columns, device)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = CustomDataset(test_dataset, columns, device)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_history = []
        test_history = []
        for epoch in range(epochs):
            start = time.time()
            print(f"Epoch {epoch}/{epochs}")
            print("Training...")
            train_losses, test_losses = [], []
            for i, (data, weights) in enumerate(train_loader):
                flow.train()
                optimizer.zero_grad()
                loss = -flow().log_prob(data)# * weights
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())
            train_history.append(np.mean(train_losses))

            print("Validating...")
            for i, (data, weights) in enumerate(val_loader):
                with torch.no_grad():
                    flow.eval()
                    loss = -flow().log_prob(data)# * weights
                    loss = loss.mean()
                    test_losses.append(loss.item())
            test_history.append(np.mean(test_losses))

            duration = time.time() - start
            print(
                f"Epoch {epoch} | Rank {device} - train loss: {train_history[epoch]:.4f} - val loss: {test_history[epoch]:.4f} - time: {duration:.2f}s"
            )
            #if early_stopper.early_stop(test_history[epoch]) or epoch == epochs - 1:
            if epoch == epochs - 1:
                print("Early stopping")
                # save the model
                # make dir if does not exist
                if not os.path.exists("flows/"):
                    os.makedirs("flows/") 
                torch.save(flow.state_dict(), f"flows/flow_{process}.pt")
                plot_loss(train_history, test_history, "flows/", process)
                evaluate_flow(flow, test_loader, "flows/", columns, process)
                break