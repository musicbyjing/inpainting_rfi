import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from models import UNet


class RFIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data, label


def prep_data(data_file, labels_file, test_size=0.2, batch_size=1):
    data, labels = np.load(data_file), np.load(labels_file)
    # Switch channels last to channels first
    data = np.moveaxis(data, -1, 2)
    labels = np.moveaxis(labels, -1, 2)
    # Convert to float, which model expects
    data = data.astype(float)
    labels = labels.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    train_dataset = RFIDataset(X_train, y_train) 
    test_dataset = RFIDataset(X_test, y_test) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # print(X.size())
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # perform inference without calculating gradients
        for X, y in dataloader:
            print("YEET", X.size())
            pred = model(X.float())
            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", flush=True)


def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    print("-------------------------------")

    train_dataloader, test_dataloader = prep_data("color_matching_exp/dataset.npy", \
                                                    "color_matching_exp/labels.npy")
                                                    # (3, 5, 4, 64, 64)

    model = UNet(model_type='deepsets', p_drop=0.1, use_max=1)
    model = model.float()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()
