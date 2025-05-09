import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from sklearn.preprocessing import LabelEncoder

from config import *
from utils import visualize_batch, plot_history


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=HAND_LENGTH, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_correct = 0, 0

        # train
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_correct += (predicted == batch_y).sum().item()

        # validate
        val_result = evaluate(model, val_loader, criterion, device)


        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_result['loss'])
        history['val_acc'].append(val_result['accuracy'])

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_result["loss"]:.4f}, Val Acc: {val_result["accuracy"]:.4f}')

    return history


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss= 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())


    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred
    }
    return results

def predict(model, input_data, device):
    model.eval()
    with torch.no_grad():
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = input_data.to(device)
        outputs = model(input_data)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
    return predicted.cpu().numpy(), probs.cpu().numpy()

class NPZDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.features = data['features']
        self.labels = data['labels']

        if isinstance(self.labels[0], str):
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(self.labels)

        self.transform = transform
        assert len(self.features) == len(self.labels), "Mismatch between features and labels"
        print(f"Dataset loaded. Total samples: {len(self.features)}")
        print(f"Feature shape: {self.features.shape}, Number of label classes: {len(np.unique(self.labels))}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y]).squeeze()
        if self.transform:
            x = self.transform(x)
        return x, y


class RandomCrop:
    def __init__(self, output_length):
        self.output_length = output_length

    def __call__(self, x):
        seq_len = x.shape[0]
        if seq_len <= self.output_length:
            return x
        start = np.random.randint(0, seq_len - self.output_length)
        return x[start:start + self.output_length, :]


def create_dataloaders(npz_path, batch_size=32, val_ratio=0.2, test_ratio=0.1):
    full_dataset = NPZDataset(npz_path)
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"\nDataset split:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

def load_model(model, path="checkpoint.pth", device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint['history']
    print(f"load model from {path} successful")
    return model, history

