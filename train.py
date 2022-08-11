import os
import glob
import argparse

from scripts.dataset import *
from scripts.engine import *
from scripts.lossFn import *
from scripts.model import *
from scripts.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", action="store", type=int, help="Epochs for Training Iteration")
parser.add_argument("--name-model", nargs="?", type=str, const="model.pth", help="Name for save a Model")
parser.add_argument("--save", action="store_true", help="Save a Model")
parser.add_argument("--early-stop", action="store_false", help="Early Stopping")
args = parser.parse_args()


def main():
    path = "data/MSRA-TD500"
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor()
        ])

    train_dataset = MSRADataset(train_path, transform=transform)
    test_dataset = MSRADataset(test_path, transform=transform)
    val_length = int(len(test_dataset) * 0.6)
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_length, len(test_dataset) - val_length])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = EAST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = EASTLoss()

    engine = Engine(model, optimizer, criterion, Epochs=args.epochs, Device=device, earlyStop=args.early_stop, save=args.save, name_model=args.name_model)
    engine.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
