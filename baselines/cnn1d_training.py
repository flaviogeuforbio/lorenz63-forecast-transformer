from copy import deepcopy 
import numpy as np 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from baselines.cnn1d_baseline import CNN1dBaseline
from torch.optim import AdamW
import torch.nn as nn
import math

def evaluate(model, loader):
    model.eval()

    #initiating arrays
    y_true = []
    y_pred = []

    #calculating predictions
    for X, rho, _ in loader:
        pred = model(X)

        y_true.append(rho.detach().numpy())
        y_pred.append(pred.detach().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    #calculating metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics, y_true, y_pred



def train_cnn1d_baseline(train_loader, val_loader, epochs = 20, lr = 3e-4, weight_decay = 1e-4, patience = 5):
    #defining model/optimizer/loss
    model = CNN1dBaseline()
    opt = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    criterion = nn.MSELoss()

    #auxiliary variables
    best_mae = float("inf")
    bad = 0
    best_state = None

    history = [] #saves train/val metrics for each epoch

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X, rho, _ in train_loader:
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, rho)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        #getting train/val metrics for each epoch
        train_loss /= len(train_loader)
        val_metrics, *_  = evaluate(model, val_loader)

        #updating history and printing results
        history.append({"epoch": epoch, "train_loss": math.sqrt(train_loss), "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"], "val_r2": val_metrics["r2"]})
        print(f"epoch: {epoch} | train_loss: {math.sqrt(train_loss):.4f} | val_mae: {val_metrics['mae']:.3f} val_rmse: {val_metrics['rmse']:.3f} val_r2: {val_metrics['r2']:.3f}")

        #checkpointing
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            bad = 0
            best_state = deepcopy(model.state_dict()) #saving a copy of model state dict

        else:
            bad += 1
            if bad >= patience:
                #triggering early stopping
                print(f"Early stopping triggered, epoch = {epoch}.")
                break
                

    #retrieving best state & evaluating final model
    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics, *_ = evaluate(model, val_loader)
    return model, best_val_metrics, history
