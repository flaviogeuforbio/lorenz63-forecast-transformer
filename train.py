import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse

from model import LorenzTransformer, LorenzTransformerConfig
from data.dataset import make_forecast_loaders
from analysis_tools import analyze_by_regime

#CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 10, help = "N. of training epochs")
parser.add_argument("--patience", type = int, default = 3, help = "Early stopping parameters (maximum n. of bad training epochs allowed)")
parser.add_argument("--checkpoint_path", help = "Path to save best model weights")
parser.add_argument("--losses_path", type = str, default = "results/training_curves.npz", help = "Path to save loss curves of the training loop")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#creating dataloaders and model
train_dl, val_dl, test_dl = make_forecast_loaders("data/lorenz_dataset.npz", num_workers = 0, pin_memory = False)

cfg = LorenzTransformerConfig()
model = LorenzTransformer(cfg).to(device)

#defining optimizer
optimizer = AdamW(model.parameters()) #default hyperparameters for now

H = cfg.H
def evaluate(model, val_dl):
    model.eval()

    #initiating cumulative losses
    total_model_loss = 0.0 
    total_pers_loss = 0.0

    #main loop
    with torch.no_grad():
        for x, y, *_ in tqdm(val_dl, leave = False, desc = "Evaluation"):
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            y_pers = x[:, -1:, :].repeat(1, H, 1)

            total_model_loss += F.mse_loss(y_pred, y).item()
            total_pers_loss += F.mse_loss(y_pers, y).item()

    model_loss = total_model_loss / len(val_dl)
    pers_loss = total_pers_loss / len(val_dl)

    return model_loss, pers_loss

#training loop
epochs = args.epochs
patience = args.patience

best_loss = float("inf")
bad = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()

    #cumulative loss
    total_loss = 0.0

    for x, y, *_ in tqdm(train_dl, leave = False, desc = f"Epoch {epoch + 1}/{epochs}"):        
        x = x.to(device)
        y = y.to(device)

        #forward & backward pass
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_dl)
    val_loss, pers_loss = evaluate(model, val_dl)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch: {epoch + 1}, train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, pers loss = {pers_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        bad = 0
        best_epoch = epoch
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.__dict__, 
            "epoch": epoch + 1, 
            "best_val": best_loss
        }, args.checkpoint_path)

    else:
        bad += 1
        if bad >= patience: #trigger early stopping
            print("Early stopping triggered.")
            break
            

#saving loss curves
if not os.path.exists("results"):
    os.makedirs("results")
np.savez(
    args.losses_path,
    train = np.array(train_losses, dtype = np.float32),
    val = np.array(val_losses, dtype = np.float32),
    best_epoch = np.array([best_epoch], dtype = np.int32)
)

