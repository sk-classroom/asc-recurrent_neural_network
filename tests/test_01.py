# %% Import
import unittest
from typing import Tuple, Optional
from torch import Tensor, zeros, eye, int64, tanh, no_grad, save, load, max
from torch.nn import Module, Linear, ReLU, Embedding, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import numpy as np
from pandas import read_csv

# %% Define utility functions
def load_model(filepath):
    # Load model parameters and state
    model_params = load(filepath)

    # Create model with saved parameters
    model = CustomNeuralNetwork(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        output_size=model_params['output_size']
    )

    # Load the state dictionary
    model.load_state_dict(model_params['state_dict'])

    return model

test_data_table = read_csv("./data/test_data.csv")
model = load_model("./assignment/trained_model.pth")
vocab_size = 34

test_target = Tensor(test_data_table["target"].values).to(int64)
test_seqs = Tensor(test_data_table.drop(columns=["target"]).values).to(int64)
test_dataset = CustomDataset(vocab_size=vocab_size, sequences=test_seqs, target=test_target)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# First calculate random baseline accuracy
random_accuracy = 100 * 1.0 / vocab_size
print(f'Random Baseline Accuracy: {random_accuracy:.2f}%')

# Evaluate model on test set
model.eval()
correct = 0
total = 0

with no_grad():
    for x_batch, y_batch in test_loader:
        predictions = model(x_batch)
        _, predicted = max(predictions, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch.squeeze()).sum().item()

test_accuracy = 100 * correct / total
print(f'Model Test Accuracy: {test_accuracy:.2f}%')
assert test_accuracy > 65.0, "Model test accuracy is less than 65.0%"

# %% Test -----------
