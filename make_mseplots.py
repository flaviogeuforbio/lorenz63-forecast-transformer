import torch
import warnings
import argparse
warnings.filterwarnings("ignore")

from analysis_tools import analyze_by_regime, plot_mse_per_step_by_regime
from model import LorenzTransformer, LorenzTransformerConfig
from data.dataset import make_forecast_loaders

#CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="checkpoints/forecast_best.pt", help = "Path to checkpoint model weights to load (.pt)")
parser.add_argument("--dataset", type=str, default="data/lorenz_dataset.npz", help = "Path to dataset file to use (.npz)")
parser.add_argument("--outpath", type=str, default="plots/mse_per_step_by_regime.png", help = "Path to save output file (.png)")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#importing model & dataloaders
cfg = LorenzTransformerConfig()
model = LorenzTransformer(cfg).to(device)
model.load_state_dict(torch.load(args.ckpt, weights_only = True)["model"]) #loading best weights

train_dl, val_dl, test_dl = make_forecast_loaders(args.dataset, return_stats = False, num_workers = 0, pin_memory = False)

#analyzing by regime model & pers. baseline predictions
perh_loss_0, perh_loss_1, perh_loss_0_pers, perh_loss_1_pers = analyze_by_regime(model, val_dl, device, pers_comparison = True)

#generating & saving plot
plot_mse_per_step_by_regime(perh_loss_0, perh_loss_0_pers, perh_loss_1, perh_loss_1_pers, outpath = args.outpath)

