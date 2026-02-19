import torch
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from analysis_tools import make_gif
from data.dataset import make_forecast_loaders
from model import LorenzTransformer, LorenzTransformerConfig
from analysis_tools import make_gif_traj, make_gif

#CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="checkpoints/forecast_best.pt", help = "Path to checkpoint model weights to load (.pt)")
parser.add_argument("--dataset", type=str, default="data/lorenz_dataset.npz", help = "Path to dataset file to use (.npz)")
parser.add_argument("--outpath", type=str, default="plots/true_vs_forecast_chaotic.gif", help = "Path to save output file (.gif)")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#importing model & dataloaders
cfg = LorenzTransformerConfig()
model = LorenzTransformer(cfg).to(device)
model.load_state_dict(torch.load(args.ckpt, weights_only = True)["model"]) #loading best weights

train_dl, val_dl, test_dl, mean, std = make_forecast_loaders(args.dataset, return_stats = True, num_workers = 0, pin_memory = False)

#creating 3D animation
H = cfg.H

#chaotic
x, y_true, y_pred, traj = make_gif_traj(
    model,
    mean,
    std,
    device,
    forecast_lenght = 300,
    rollout_stride = 8, 
    stop_idx = 5000      
)
os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
make_gif(x, y_true, y_pred, traj, outpath = args.outpath)